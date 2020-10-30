import ArgumentParser
import BeeDataset
import BeeTracking
import PythonKit
import SwiftFusion
import TensorFlow

struct BeeTrackingTool: ParsableCommand {
  static var configuration = CommandConfiguration(
    subcommands: [TrainRAE.self, InferTrack.self])
}

/// The dimension of the hidden layer in the appearance model.
let kHiddenDimension = 500

/// The dimension of the latent code in the appearance model.
let kLatentDimension = 10

/// Returns a `[N, h, w, c]` batch of normalized frames from a bee video, and returns the statistics
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

struct TrainRAE: ParsableCommand {
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

struct InferTrack: ParsableCommand {
  @Option(help: "Load weights from this file")
  var loadWeights: String

  func run() {
    let np = Python.import("numpy")

    let video = BeeVideo(videoName: "bee_video_1")!
    let (beeBatch, frameStatistics) = makeBeeBatch()
    let (imageHeight, imageWidth, imageChannels) =
      (beeBatch.shape[1], beeBatch.shape[2], beeBatch.shape[3])

    var model = DenseRAE(
      imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels,
      hiddenDimension: kHiddenDimension, latentDimension: kLatentDimension)
    model.load(weights: np.load(loadWeights, allow_pickle: true))

    let tracker = TrackingFactorGraph(
      model, video, frameStatistics, trackId: 0, indexStart: 0, length: 10)
    var v = tracker.v

    var optimizer = LM(precision: 1e1, max_iteration: 400)
    optimizer.verbosity = .SUMMARY
    try? optimizer.optimize(graph: tracker.fg, initial: &v)
  }
}

BeeTrackingTool.main()
