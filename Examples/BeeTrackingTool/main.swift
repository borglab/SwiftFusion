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
    subcommands: [TrainRAE.self, InferTrackRAE.self, InferTrackRawPixels.self, NaiveRae.self])
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

    print(boxes.count)
  }
}

/// Infers a track on a VOT video, using the raw pixel tracker.
struct InferTrackRawPixels: ParsableCommand {
  func run() {

    func rawPixelTracker(_ frames: [Tensor<Float>], _ start: OrientedBoundingBox) -> [OrientedBoundingBox] {
      var tracker = makeRawPixelTracker(frames: frames, target: frames[0].patch(at: start))
      tracker.optimizer.precision = 1e0
      let prediction = tracker.infer(knownStart: Tuple1(start.center))
      return tracker.frameVariableIDs.map { varIds in
        let poseId = varIds.head
        return OrientedBoundingBox(center: prediction[poseId], rows: start.rows, cols: start.cols)
      }
    }

    var dataset = OISTBeeVideo()!
    // Only do inference on the interesting tracks.
    dataset.tracks = [3, 5, 6, 7].map { dataset.tracks[$0] }
    let trackerEvaluationDataset = TrackerEvaluationDataset(dataset)
    let eval = trackerEvaluationDataset.evaluate(
      rawPixelTracker, sequenceCount: dataset.tracks.count, deltaAnchor: 100, outputFile: "rawpixel.json")
    print(eval.trackerMetrics.accuracy)
    print(eval.trackerMetrics.robustness)
  }
}

/// Tracking with a Naive Bayes with RAE
struct NaiveRae: ParsableCommand {
  @Option(help: "Where to load the RAE weights")
  var loadWeights: String

  @Option(help: "The dimension of the latent code in the RAE appearance model")
  var kLatentDimension: Int

  @Option(help: "The dimension of the hidden code in the RAE appearance model")
  var kHiddenDimension = 100

  @Flag
  var verbose: Bool = false

  @Option
  var outputFile: String

  @Option
  var truncate: Int

  /// Returns predictions for `videoName` using the raw pixel tracker.
  func naiveRaeTrack(dataset dataset_: OISTBeeVideo) {
    var dataset = dataset_
    dataset.labels = dataset.labels.map {
      $0.filter({ $0.label == .Body })
    }
    // Make batch and do RAE
    let (batch, _) = dataset.makeBatch(appearanceModelSize: (40, 70), batchSize: 200)
    var statistics = FrameStatistics(batch)
    statistics.mean = Tensor(62.26806976644069)
    statistics.standardDeviation = Tensor(37.44683834503672)

    let backgroundBatch = dataset.makeBackgroundBatch(
      patchSize: (40, 70), appearanceModelSize: (40, 70),
      statistics: statistics,
      batchSize: 300
    )

    let (imageHeight, imageWidth, imageChannels) =
      (batch.shape[1], batch.shape[2], batch.shape[3])
    
    if verbose { print("Loading RAE model, \(batch.shape)...") }
    
    let np = Python.import("numpy")

    var rae = DenseRAE(
      imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels,
      hiddenDimension: kHiddenDimension, latentDimension: kLatentDimension
    )
    rae.load(weights: np.load(loadWeights, allow_pickle: true))

    if verbose { print("Fitting Naive Bayes model") }

    var (foregroundModel, backgroundModel) = (
      MultivariateGaussian(
        dims: TensorShape([kLatentDimension]),
        regularizer: 1e-3
      ), GaussianNB(
        dims: TensorShape([kLatentDimension]),
        regularizer: 1e-3
      )
    )

    let batchPositive = rae.encode(batch)
    foregroundModel.fit(batchPositive)

    let batchNegative = rae.encode(backgroundBatch)
    backgroundModel.fit(batchNegative)

    if verbose {
      print("Foreground: \(foregroundModel)")
      print("Background: \(backgroundModel)")
    }

    func tracker(_ frames: [Tensor<Float>], _ start: OrientedBoundingBox) -> [OrientedBoundingBox] {
      var tracker = makeNaiveBayesAETracker(
        model: rae,
        statistics: statistics,
        frames: frames,
        targetSize: (start.rows, start.cols),
        foregroundModel: foregroundModel, backgroundModel: backgroundModel
      )
      tracker.optimizer.cgls_precision = 1e-5
      tracker.optimizer.precision = 1e-3
      tracker.optimizer.max_iteration = 200
      let prediction = tracker.infer(knownStart: Tuple1(start.center))
      return tracker.frameVariableIDs.map { varIds in
        let poseId = varIds.head
        return OrientedBoundingBox(center: prediction[poseId], rows: start.rows, cols: start.cols)
      }
    }

    // Only do inference on the interesting tracks.
    var evalDataset = OISTBeeVideo(truncate: truncate)!
    evalDataset.tracks = [3, 5, 6, 7].map { evalDataset.tracks[$0] }
    let trackerEvaluationDataset = TrackerEvaluationDataset(evalDataset)
    let eval = trackerEvaluationDataset.evaluate(
      tracker, sequenceCount: evalDataset.tracks.count, deltaAnchor: 500, outputFile: outputFile)
    print(eval.trackerMetrics.accuracy)
    print(eval.trackerMetrics.robustness)
  }

  func run() {
    if verbose {
      print("Loading dataset...")
    }

    startTimer("DATASET_LOAD")
    let dataset: OISTBeeVideo = OISTBeeVideo(deferLoadingFrames: true)!
    stopTimer("DATASET_LOAD")

    if verbose {
      print("Tracking...")
    }

    startTimer("RAE_TRACKING")
    naiveRaeTrack(dataset: dataset)
    stopTimer("RAE_TRACKING")

    if verbose {
      printTimers()
    }

    print(boxes.count)
  }
}

// It is important to set the global threadpool before doing anything else, so that nothing
// accidentally uses the default threadpool.
ComputeThreadPools.global =
  NonBlockingThreadPool<PosixConcurrencyPlatform>(name: "mypool", threadCount: 12)

BeeTrackingTool.main()
