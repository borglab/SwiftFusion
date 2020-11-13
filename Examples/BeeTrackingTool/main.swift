import ArgumentParser
import BeeDataset
import BeeTracking
import PenguinParallelWithFoundation
import PenguinStructures
import PythonKit
import SwiftFusion
import TensorFlow

struct BeeTrackingTool: ParsableCommand {
  static var configuration = CommandConfiguration(
    subcommands: [TrainRAE.self, InferTrackRAE.self, InferTrackRawPixels.self])
}

/// The dimension of the hidden layer in the appearance model.
let kHiddenDimension = 500

/// The dimension of the latent code in the appearance model.
let kLatentDimension = 10

/// Returns a `[N, h, w, c]` batch of normalized patches from a VOT video, and returns the
/// statistics used to normalize them.
func makeVOTBatch(votBaseDirectory: String, videoName: String, appearanceModelSize: (Int, Int))
  -> (normalized: Tensor<Double>, statistics: FrameStatistics)
{
  let data = VOTVideo(votBaseDirectory: votBaseDirectory, videoName: videoName)!
  let images = (0..<data.frames.count).map { (i: Int) -> Tensor<Double> in
    return Tensor<Double>(data.frames[i].patch(at: data.track[i], outputSize: appearanceModelSize))
  }
  let stacked = Tensor(stacking: images)
  let statistics = FrameStatistics(stacked)
  return (statistics.normalized(stacked), statistics)
}

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

  @Option(help: "Number of rows in the appearance model output")
  var appearanceModelRows: Int = 100

  @Option(help: "Number of columns in the appearance model output")
  var appearanceModelCols: Int = 100

  func run() {
    let np = Python.import("numpy")

    let (batch, _) = makeVOTBatch(
      votBaseDirectory: votBaseDirectory, videoName: videoName, appearanceModelSize: (appearanceModelRows, appearanceModelCols))
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

/// Infers a track on a VOT video, using the RAE tracker.
struct InferTrackRAE: ParsableCommand {
  @Option(help: "Load weights from this file")
  var loadWeights: String

  @Option(help: "Base directory of the VOT dataset")
  var votBaseDirectory: String

  @Option(help: "Name of the VOT video to use")
  var videoName: String

  @Option(help: "Number of rows in the appearance model output")
  var appearanceModelRows: Int = 100

  @Option(help: "Number of columns in the appearance model output")
  var appearanceModelCols: Int = 100

  @Option(help: "How many frames to track")
  var frameCount: Int = 50

  @Flag(help: "Print progress information")
  var verbose: Bool = false

  func run() {
    let np = Python.import("numpy")

    let appearanceModelSize = (appearanceModelRows, appearanceModelCols)

    let video = VOTVideo(votBaseDirectory: votBaseDirectory, videoName: videoName)!
    let (_, frameStatistics) = makeVOTBatch(
      votBaseDirectory: votBaseDirectory, videoName: videoName,
      appearanceModelSize: appearanceModelSize)
    var model = DenseRAE(
      imageHeight: appearanceModelRows, imageWidth: appearanceModelCols,
      imageChannels: video.frames[0].shape[2],
      hiddenDimension: kHiddenDimension, latentDimension: kLatentDimension)
    model.load(weights: np.load(loadWeights, allow_pickle: true))

    let videoSlice = video[0..<min(video.frames.count, frameCount)]

    var tracker = makeRAETracker(
      model: model,
      statistics: frameStatistics,
      frames: videoSlice.frames,
      targetSize: (video.track[0].rows, video.track[0].cols))

    if verbose { tracker.optimizer.verbosity = .SUMMARY }

    let startPose = videoSlice.track[0].center
    let startPatch = Tensor<Double>(videoSlice.frames[0].patch(
      at: videoSlice.track[0], outputSize: appearanceModelSize))
    let startLatent = Vector10(
      flatTensor: model.encode(
        frameStatistics.normalized(startPatch).expandingShape(at: 0)).squeezingShape(at: 0))
    let prediction = tracker.infer(knownStart: Tuple2(startPose, startLatent))

    let boxes = tracker.frameVariableIDs.map { frameVariableIDs -> OrientedBoundingBox in
      let poseID = frameVariableIDs.head
      return OrientedBoundingBox(
        center: prediction[poseID], rows: video.track[0].rows, cols: video.track[0].cols)
    }
  }
}

/// Infers a track on a VOT video, using the raw pixel tracker.
struct InferTrackRawPixels: ParsableCommand {
  @Option(help: "Base directory of the VOT dataset")
  var votBaseDirectory: String

  @Option(help: "Name of the VOT video to use")
  var videoName: String

  @Option(help: "How many frames to track")
  var frameCount: Int = 50

  @Flag(help: "Print progress information")
  var verbose: Bool = false

  func run() {
    let dataset = OISTBeeVideo()!
    let trackerEvaluationDataset = TrackerEvaluationDataset(dataset)
    func rawPixelTracker(_ frames: [Tensor<Float>], _ start: OrientedBoundingBox) -> [OrientedBoundingBox] {
      var tracker = makeRawPixelTracker(frames: frames, target: frames[0].patch(at: start))
      tracker.optimizer.precision = 1e0
      let prediction = tracker.infer(knownStart: Tuple1(start.center))
      return tracker.frameVariableIDs.map { varIds in
	let poseId = varIds.head
	return OrientedBoundingBox(center: prediction[poseId], rows: start.rows, cols: start.cols)
      }
    }
    let eval = trackerEvaluationDataset.evaluate(rawPixelTracker, sequenceCount: 20)
    print(eval.trackerMetrics.robustness)
  }
}

// It is important to set the global threadpool before doing anything else, so that nothing
// accidentally uses the default threadpool.
ComputeThreadPools.global =
  NonBlockingThreadPool<PosixConcurrencyPlatform>(name: "mypool", threadCount: 12)

BeeTrackingTool.main()
