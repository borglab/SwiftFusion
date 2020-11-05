import ArgumentParser
import BeeDataset
import BeeTracking
import PenguinParallelWithFoundation
import PythonKit
import SwiftFusion
import TensorFlow

struct BeeTrackingTool: ParsableCommand {
  static var configuration = CommandConfiguration(
    subcommands: [TrainRAE.self, InferTrack.self, TrainBeeRAE.self, InferBeeTrack.self])
}

/// The dimension of the hidden layer in the appearance model.
let kHiddenDimension = 500

/// The dimension of the latent code in the appearance model.
let kLatentDimension = 10

/// Returns a `[N, h, w, c]` batch of normalized patches from a bee video, and returns the statistics
/// used to normalize them.
func makeBeeBatch() -> (normalized: Tensor<Double>, statistics: FrameStatistics) {
  let data = BeeVideo(videoName: "bee_video_1")
  let num_samples = 80
  let images = (0..<num_samples).map { (i: Int) -> Tensor<Double> in
    return data!.frames[i].patch(at: data!.tracks[1][i].location)
  }
  let stacked = Tensor(stacking: images)
  let statistics = FrameStatistics(stacked)
  return (statistics.normalized(stacked), statistics)
}

/// Returns a `[N, h, w, c]` batch of normalized patches from a VOT video, and returns the
/// statistics used to normalize them.
func makeVOTBatch(votBaseDirectory: String, videoName: String, patchSize: (Int, Int))
  -> (normalized: Tensor<Double>, statistics: FrameStatistics)
{
  let data = VOTVideo(votBaseDirectory: votBaseDirectory, videoName: videoName)!
  let images = (0..<data.frames.count).map { (i: Int) -> Tensor<Double> in
    return data.frames[i].patch(at: data.track[i], outputSize: patchSize)
  }
  let stacked = Tensor(stacking: images)
  let statistics = FrameStatistics(stacked)
  return (statistics.normalized(stacked), statistics)
}

// How this is going to work.
// So I need the thing to support differentiable scaling first.
// Then we can choose a size, scale all the points to that size, train the RAE on that.
// Now there is a variable for the size that is used in the factor. The graph magically works!!

/// Trains a RAE on the VOT dataset.
struct TrainRAE: ParsableCommand {
  @Option(help: "Load weights from this file before training")
  var loadWeights: String?

  @Option(help: "Save weights to this file after training")
  var saveWeights: String

  @Option(help: "Number of epochs to train")
  var epochCount: Int = 20

  @Option(help: "Base directory of the VOT dataset")
  var votBaseDirectory: String

  @Option(help: "Name of the VOT video to use")
  var videoName: String

  @Option(help: "Number of rows to scale the patch to")
  var patchRows: Int = 100

  @Option(help: "Number of columns to scale the patch to")
  var patchCols: Int = 100

  func run() {
    let np = Python.import("numpy")

    let (batch, _) = makeVOTBatch(
      votBaseDirectory: votBaseDirectory, videoName: videoName, patchSize: (patchRows, patchCols))
    print("Batch shape: \(batch.shape)")

    let (imageHeight, imageWidth, imageChannels) =
      (batch.shape[1], batch.shape[2], batch.shape[3])

    var model = DenseRAE(
      imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels,
      hiddenDimension: kHiddenDimension, latentDimension: kLatentDimension)
    if let loadWeights = loadWeights {
      let weights = np.load(loadWeights, allow_pickle: true)
      model.load(weights: weights)
    }

    let loss = DenseRAELoss()
    _ = loss(model, batch, printLoss: true)

    // Use ADAM as optimizer
    let optimizer = Adam(for: model)
    optimizer.learningRate = 1e-3

    // Thread-local variable that model layers read to know their mode
    Context.local.learningPhase = .training

    for i in 0..<epochCount {
      print("Step \(i), loss: \(loss(model, batch))")

      let grad = gradient(at: model) { loss($0, batch) }
      optimizer.update(&model, along: grad)
    }

    _ = loss(model, batch, printLoss: true)

    np.save(saveWeights, np.array(model.numpyWeights, dtype: Python.object))
  }
}

/// Infers a track on a VOT video.
struct InferTrack: ParsableCommand {
  @Option(help: "Load weights from this file")
  var loadWeights: String

  @Option(help: "Base directory of the VOT dataset")
  var votBaseDirectory: String

  @Option(help: "Name of the VOT video to use")
  var videoName: String

  @Option(help: "Number of rows to scale the patch to")
  var patchRows: Int = 100

  @Option(help: "Number of columns to scale the patch to")
  var patchCols: Int = 100

  func run() {
    // ComputeThreadPools.global = NonBlockingThreadPool<PosixConcurrencyPlatform>(name: "mypool", threadCount: 12)

    // let np = Python.import("numpy")

    // let video = VOTVideo(votBaseDirectory: votBaseDirectory, videoName: videoName)!
    // let (batch, frameStatistics) = makeVOTBatch(
    //   votBaseDirectory: votBaseDirectory, videoName: videoName, patchSize: (patchRows, patchCols))
    // let (imageHeight, imageWidth, imageChannels) =
    //   (batch.shape[1], batch.shape[2], batch.shape[3])

    // var model = DenseRAE(
    //   imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels,
    //   hiddenDimension: kHiddenDimension, latentDimension: kLatentDimension)
    // model.load(weights: np.load(loadWeights, allow_pickle: true))

    // let tracker = TrackingFactorGraph(
    //   model, video, frameStatistics, indexStart: 0, length: 10, patchSize: (patchRows, patchCols))
    // var v = tracker.v

    // startTimer("optimize")
    // var optimizer = LM(precision: 1e1, max_iteration: 400)
    // optimizer.verbosity = .SUMMARY
    // optimizer.cgls_precision = 1e-6
    // try? optimizer.optimize(graph: tracker.fg, initial: &v)
    // stopTimer("optimize")

    // printTimers()
    // printCounters()
  }
}

/// Trains a RAE on the bee dataset.
struct TrainBeeRAE: ParsableCommand {
  @Option(help: "Load weights from this file before training")
  var loadWeights: String?

  @Option(help: "Save weights to this file after training")
  var saveWeights: String

  @Option(help: "Number of epochs to train")
  var epochCount: Int = 20

  func run() {
    let np = Python.import("numpy")

    let (beeBatch, _) = makeBeeBatch()
    print("Bee batch shape: \(beeBatch.shape)")

    let (imageHeight, imageWidth, imageChannels) =
      (beeBatch.shape[1], beeBatch.shape[2], beeBatch.shape[3])

    var model = DenseRAE(
      imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels,
      hiddenDimension: kHiddenDimension, latentDimension: kLatentDimension)
    if let loadWeights = loadWeights {
      let weights = np.load(loadWeights, allow_pickle: true)
      model.load(weights: weights)
    }

    let loss = DenseRAELoss()
    _ = loss(model, beeBatch, printLoss: true)

    // Use ADAM as optimizer
    let optimizer = Adam(for: model)
    optimizer.learningRate = 1e-3

    // Thread-local variable that model layers read to know their mode
    Context.local.learningPhase = .training

    for i in 0..<epochCount {
      print("Step \(i), loss: \(loss(model, beeBatch))")

      let grad = gradient(at: model) { loss($0, beeBatch) }
      optimizer.update(&model, along: grad)
    }

    _ = loss(model, beeBatch, printLoss: true)

    np.save(saveWeights, np.array(model.numpyWeights, dtype: Python.object))
  }
}

/// Infers a track on a bee video.
struct InferBeeTrack: ParsableCommand {
  @Option(help: "Load weights from this file")
  var loadWeights: String

  func run() {
    ComputeThreadPools.global = NonBlockingThreadPool<PosixConcurrencyPlatform>(name: "mypool", threadCount: 12)

    let np = Python.import("numpy")

    let video = BeeVideo(videoName: "bee_video_1")!
    let (beeBatch, frameStatistics) = makeBeeBatch()
    let (imageHeight, imageWidth, imageChannels) =
      (beeBatch.shape[1], beeBatch.shape[2], beeBatch.shape[3])

    var model = DenseRAE(
      imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels,
      hiddenDimension: kHiddenDimension, latentDimension: kLatentDimension)
    model.load(weights: np.load(loadWeights, allow_pickle: true))

    let tracker = AppearanceTrackingFactorGraph(
      model, video, frameStatistics, trackId: 0, indexStart: 0, length: 10)
    var v = tracker.v

    startTimer("optimize")
    var optimizer = LM(precision: 1e1, max_iteration: 400)
    optimizer.verbosity = .SUMMARY
    optimizer.cgls_precision = 1e-6
    try? optimizer.optimize(graph: tracker.fg, initial: &v)
    stopTimer("optimize")

    printTimers()
    printCounters()
  }
}

BeeTrackingTool.main()
