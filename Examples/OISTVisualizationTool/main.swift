// Copyright 2020 The SwiftFusion Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import ArgumentParser
import BeeDataset
import BeeTracking
import PenguinStructures
import PenguinParallelWithFoundation
import PythonKit
import SwiftFusion
import TensorFlow
import Foundation

struct OISTVisualizationTool: ParsableCommand {
  static var configuration = CommandConfiguration(
    subcommands: [VisualizeTrack.self, ViewFrame.self, RawTrack.self, PpcaTrack.self, NaiveRae.self, TrainRAE.self, NaivePca.self])
}

/// View a frame with bounding boxes
struct ViewFrame: ParsableCommand {
  @Option(help: "Location of dataset folder which should contain `frames` and `frames_txt`")
  var datasetLocation: String = "./OIST_Data"

  @Option(help: "Which frame to show")
  var frameId: Int = 0

  func run() {
    let dataURL = URL(fileURLWithPath: datasetLocation)
    print("Viewing \(dataURL) at frame \(frameId)")
    let dataset = OISTBeeVideo(directory: dataURL, deferLoadingFrames: true)!

    let frameRawId = dataset.frameIds[frameId]

    let image = dataset.loadFrame(frameRawId)!

    // plot(image, boxes: dataset.labels[frameId].enumerated().map {
    //   (String($0), $1.location)
    // }, margin: 10.0, scale: 0.5).show()
  }
}

/// Returns a `[N, h, w, c]` batch of normalized patches from a VOT video, and returns the
/// statistics used to normalize them.
/// - dataset: Bee video dataset object
/// - appearanceModelSize: [H, W]
/// - batchSize: number of batch samples
/// - seed: Allow controlling the random sequence
/// - trainSplit: Controls where in the frames to split between train and test
func makeOISTBatch(dataset: OISTBeeVideo, appearanceModelSize: (Int, Int), batchSize: Int = 300, seed: Int = 42, trainSplit: Int = 250)
  -> (normalized: Tensor<Double>, statistics: FrameStatistics)
{
  var images: [Tensor<Double>] = []
  images.reserveCapacity(batchSize)

  var currentFrame: Tensor<Double> = [0]
  var currentId: Int = -1

  var deterministicEntropy = ARC4RandomNumberGenerator(seed: seed)
  for label in dataset.labels[0..<trainSplit].randomSelectionWithoutReplacement(k: 10, using: &deterministicEntropy).lazy.joined().randomSelectionWithoutReplacement(k: batchSize, using: &deterministicEntropy).sorted(by: { $0.frameIndex < $1.frameIndex }) {
    if currentId != label.frameIndex {
      currentFrame = Tensor<Double>(dataset.loadFrame(label.frameIndex)!)
      currentId = label.frameIndex
    }
    images.append(currentFrame.patch(at: label.location, outputSize: appearanceModelSize))
  }

  let stacked = Tensor(stacking: images)
  let statistics = FrameStatistics(stacked)
  return (statistics.normalized(stacked), statistics)
}

/// Tracking with a raw l2 error
struct RawTrack: ParsableCommand {
  @Option(help: "Location of dataset folder which should contain `frames` and `frames_txt`")
  var datasetLocation: String = "./OIST_Data"

  @Option(help: "Which bounding box to track")
  var boxId: Int = 0

  @Option(help: "Track for how many frames")
  var trackFrames: Int = 10

  @Flag(help: "Print progress information")
  var verbose: Bool = false


  /// Returns predictions for `videoName` using the raw pixel tracker.
  func rawPixelTrack(dataset: OISTBeeVideo, length: Int) -> [OrientedBoundingBox] {
    // Load the video and take a slice of it.
    let videos = (0..<length).map { (i) -> Tensor<Float> in
      if verbose {
        print(".", terminator: "")
      }
      return withDevice(.cpu) { dataset.loadFrame(dataset.frameIds[i])! }
    }

    if verbose {
      print("")
    }

    let startPatch = videos[0].patch(at: dataset.labels[0][boxId].location)
    let startPose = dataset.labels[0][boxId].location.center

    if verbose {
      print("Creating tracker, startPose = \(startPose)")
    }
    
    var tracker = makeRawPixelTracker(frames: videos, target: startPatch)

    // if verbose { tracker.optimizer.verbosity = .SUMMARY } For LM Optimizer

    let prediction = tracker.infer(knownStart: Tuple1(startPose))

    let boxes = tracker.frameVariableIDs.map { frameVariableIDs -> OrientedBoundingBox in
      let poseID = frameVariableIDs.head
      return OrientedBoundingBox(
        center: prediction[poseID], rows: dataset.labels[0][boxId].location.rows, cols: dataset.labels[0][boxId].location.cols)
    }

    return boxes
  }

  func run() {
    let dataURL = URL(fileURLWithPath: datasetLocation)

    startTimer("DATASET_LOAD")
    let dataset = OISTBeeVideo(directory: dataURL, deferLoadingFrames: true)!
    stopTimer("DATASET_LOAD")

    startTimer("RAW_TRACKING")
    var bboxes: [OrientedBoundingBox]
    bboxes = rawPixelTrack(dataset: dataset, length: trackFrames)
    stopTimer("RAW_TRACKING")

    let frameRawId = dataset.frameIds[trackFrames]
    let image = dataset.loadFrame(frameRawId)!

    if verbose {
      print("Creating output plot")
    }
    startTimer("PLOTTING")
    // plot(image, boxes: bboxes.indices.map {
    //   ("\($0)", bboxes[$0])
    // }, margin: 10.0, scale: 0.5).show()
    stopTimer("PLOTTING")

    if verbose {
      printTimers()
    }
  }
}

/// Tracking with a PPCA graph
struct PpcaTrack: ParsableCommand {
  @Option(help: "Location of dataset folder which should contain `frames` and `frames_txt`")
  var datasetLocation: String = "./OIST_Data"

  @Option(help: "Which bounding box to track")
  var boxId: Int = 0

  @Option(help: "Track for how many frames")
  var trackFrames: Int = 10

  @Flag(help: "Print progress information")
  var verbose: Bool = false

  /// Returns predictions for `videoName` using the raw pixel tracker.
  func ppcaTrack(dataset dataset_: OISTBeeVideo, length: Int, ppcaSize: Int = 10, ppcaSamples: Int = 100) -> [OrientedBoundingBox] {
    var dataset = dataset_
    dataset.labels = dataset.labels.map {
      $0.filter({ $0.label == .Body })
    }
    // Make batch and do PPCA
    let (batch, statistics) = makeOISTBatch(dataset: dataset, appearanceModelSize: (40, 70))

    if verbose { print("Training PPCA model, \(batch.shape)...") }
    var ppca = PPCA(latentSize: ppcaSize)
    ppca.train(images: batch)

    if verbose { print("Loading video frames...") }
    startTimer("VIDEO_LOAD")
    // Load the video and take a slice of it.
    let videos = (0..<length).map { (i) -> Tensor<Float> in
      return withDevice(.cpu) { dataset.loadFrame(dataset.frameIds[i])! }
    }
    stopTimer("VIDEO_LOAD")

    let startPatch = statistics.normalized(videos[0].patch(at: dataset.labels[0][boxId].location))
    let startPose = dataset.labels[0][boxId].location.center
    let startLatent = ppca.encode(Tensor<Double>(startPatch))

    if verbose {
      print("Creating tracker, startPose = \(startPose)")
    }
    
    startTimer("MAKE_GRAPH")
    var tracker = makePPCATracker(model: ppca, statistics: statistics, frames: videos, targetSize: (40, 70))
    stopTimer("MAKE_GRAPH")
    // For LM Optimizer
    // if verbose { tracker.optimizer.verbosity = .SUMMARY }

    // tracker.optimizer.cgls_precision = 1e-6
    // tracker.optimizer.precision = 1e-2

    startTimer("GRAPH_INFER")
    let prediction = tracker.infer(knownStart: Tuple2(startPose, Vector10(flatTensor: startLatent)))
    stopTimer("GRAPH_INFER")

    let boxes = tracker.frameVariableIDs.map { frameVariableIDs -> OrientedBoundingBox in
      let poseID = frameVariableIDs.head
      return OrientedBoundingBox(
        center: prediction[poseID], rows: dataset.labels[0][boxId].location.rows, cols: dataset.labels[0][boxId].location.cols)
    }

    return boxes
  }

  func run() {
    let dataURL = URL(fileURLWithPath: datasetLocation)

    if verbose {
      print("Loading dataset...")
    }
    let dataset: OISTBeeVideo = { () -> OISTBeeVideo in
      startTimer("DATASET_LOAD")
      return OISTBeeVideo(directory: dataURL, deferLoadingFrames: true)!
    }()

    stopTimer("DATASET_LOAD")

    if verbose {
      print("Tracking...")
    }

    startTimer("PPCA_TRACKING")
    var bboxes: [OrientedBoundingBox]
    bboxes = ppcaTrack(dataset: dataset, length: trackFrames)
    stopTimer("PPCA_TRACKING")

    let frameRawId = dataset.frameIds[trackFrames]
    let image = dataset.loadFrame(frameRawId)!

    if verbose {
      print("Creating output plot")
    }
    startTimer("PLOTTING")
    // plot(image, boxes: bboxes.indices.map {
    //   ("\($0)", bboxes[$0])
    // }, margin: 10.0, scale: 0.5).show()
    stopTimer("PLOTTING")

    if verbose {
      printTimers()
    }
  }
}

/// The dimension of the hidden layer in the RAE appearance model.
let kHiddenDimension = 100

/// Tracking with a Naive Bayes with RAE
struct NaiveRae: ParsableCommand {
  @Option(help: "Where to load the RAE weights")
  var loadWeights: String = "./oist_rae_weight.npy"

  @Option(help: "Which bounding box to track")
  var boxId: Int = 0

  @Option(help: "Track for how many frames")
  var trackFrames: Int = 10

  @Option(help: "Track the target from frame x")
  var trackStartFrame: Int = 250

  @Option(help: "The dimension of the latent code in the RAE appearance model")
  var kLatentDimension = 10

  @Flag(help: "Print progress information")
  var verbose: Bool = false

  /// Returns predictions for `videoName` using the raw pixel tracker.
  func naiveRaeTrack(dataset dataset_: OISTBeeVideo, length: Int, startFrom: Int) -> [OrientedBoundingBox] {
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

    let batchPositive = rae.encode(batch)
    let foregroundModel = MultivariateGaussian(from:batchPositive, regularizer: 1e-3)

    let batchNegative = rae.encode(backgroundBatch)
    let backgroundModel = GaussianNB(from: batchNegative, regularizer: 1e-3)

    if verbose {
      print("Foreground: \(foregroundModel)")
      print("Background: \(backgroundModel)")
    }

    if verbose { print("Loading video frames...") }
    startTimer("VIDEO_LOAD")
    // Load the video and take a slice of it.
    let videos = (0..<length).map { (i) -> Tensor<Float> in
      return withDevice(.cpu) { dataset.loadFrame(dataset.frameIds[startFrom + i])! }
    }
    stopTimer("VIDEO_LOAD")

    let startPose = dataset.labels[startFrom][boxId].location.center

    if verbose {
      print("Creating tracker, startPose = \(startPose)")
    }
    
    startTimer("MAKE_GRAPH")
    var tracker = makeNaiveBayesAETracker(
      model: rae,
      statistics: statistics,
      frames: videos,
      targetSize: (dataset.labels[startFrom][boxId].location.rows, dataset.labels[startFrom][boxId].location.cols),
      foregroundModel: foregroundModel, backgroundModel: backgroundModel
    )
    stopTimer("MAKE_GRAPH")

    if verbose { print("Starting Optimization...") }
    // For LM Optimizer
    // if verbose { tracker.optimizer.verbosity = .SUMMARY }

    // tracker.optimizer.cgls_precision = 1e-7
    // tracker.optimizer.precision = 1e-4
    // tracker.optimizer.max_iteration = 200

    startTimer("GRAPH_INFER")
    let prediction = tracker.infer(knownStart: Tuple1(startPose))
    stopTimer("GRAPH_INFER")

    let boxes = tracker.frameVariableIDs.map { frameVariableIDs -> OrientedBoundingBox in
      let poseID = frameVariableIDs.head
      return OrientedBoundingBox(
        center: prediction[poseID], rows: dataset.labels[startFrom][boxId].location.rows, cols: dataset.labels[startFrom][boxId].location.cols)
    }

    return boxes
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
    var bboxes: [OrientedBoundingBox]
    bboxes = naiveRaeTrack(dataset: dataset, length: trackFrames, startFrom: trackStartFrame)
    stopTimer("RAE_TRACKING")

    let frameRawId = dataset.frameIds[trackStartFrame + trackFrames]
    let image = dataset.loadFrame(frameRawId)!

    if verbose {
      print("Creating output plot")
    }
    startTimer("PLOTTING")
    // plot(image, boxes: bboxes.indices.map {
    //   ("\($0)", bboxes[$0])
    // }, margin: 10.0, scale: 0.5).show()
    stopTimer("PLOTTING")

    if verbose {
      printTimers()
    }
  }
}

/// Trains a RAE on the VOT dataset.
struct TrainRAE: ParsableCommand {
  @Option(help: "Load weights from this file before training")
  var loadWeights: String?

  @Option(help: "Save weights to this file after training")
  var saveWeights: String = "./oist_rae_weight"

  @Option(help: "Number of iterations for each epoch")
  var iterationCount: Int = 300

  @Option(help: "Number of epochs")
  var epochCount: Int = 200

  @Option(help: "Number of rows in the appearance model output")
  var appearanceModelRows: Int = 40

  @Option(help: "Number of columns in the appearance model output")
  var appearanceModelCols: Int = 70

  @Option(help: "The dimension of the latent code in the RAE appearance model")
  var kLatentDimension = 10

  func run() {
    let np = Python.import("numpy")

    let dataset: OISTBeeVideo = { () -> OISTBeeVideo in
      startTimer("DATASET_LOAD")
      return OISTBeeVideo(directory: URL(fileURLWithPath: "./OIST_Data"), deferLoadingFrames: true)!
    }()

    stopTimer("DATASET_LOAD")
    
    let (bundle, statistics) = makeOISTTrainingBatch(dataset: dataset, appearanceModelSize: (40, 70), batchSize: 20000, seed: Int.random(in: 0..<9999999))
    print("Dataset size: \(bundle.count)")
    print("Statistics: \(statistics)")
    let (imageHeight, imageWidth, imageChannels) =
      (bundle[0].shape[0], bundle[0].shape[1], bundle[0].shape[2])

    var model = DenseRAE(
      imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels,
      hiddenDimension: kHiddenDimension, latentDimension: kLatentDimension)
    if let loadWeights = loadWeights {
      let weights = np.load(loadWeights, allow_pickle: true)
      model.load(weights: weights)
    }

    let loss = DenseRAELoss()
    // _ = loss(model, batch, printLoss: true)

    // Use ADAM as optimizer
    let optimizer = Adam(for: model)
    optimizer.learningRate = 1e-3

    // Thread-local variable that model layers read to know their mode
    Context.local.learningPhase = .training

    let epochs = TrainingEpochs(samples: bundle, batchSize: 200)
    var trainLossResults: [Double] = []
    for (epochIndex, epoch) in epochs.prefix(epochCount).enumerated() {
      var epochLoss: Double = 0
      var batchCount: Int = 0
      // epoch is a Slices object, see below
      for batchSamples in epoch {
        let batch = batchSamples.collated
        let (loss, grad) = valueWithGradient(at: model) { loss($0, batch) }
        optimizer.update(&model, along: grad)
        epochLoss += loss.scalarized()
        batchCount += 1
      }
      epochLoss /= Double(batchCount)
      trainLossResults.append(epochLoss)
      if epochIndex % 50 == 0 {
          print("Epoch \(epochIndex): Loss: \(epochLoss)")
      }
    }

    _ = loss(model, Tensor<Double>(stacking: bundle), printLoss: true)

    np.save(saveWeights, np.array(model.numpyWeights, dtype: Python.object))
  }
}

/// PPCA with probabilistic model
///
/// Tracking with a Naive Bayes with RAE
struct NaivePca: ParsableCommand {
  @Option(help: "Where to load the RAE weights")
  var loadWeights: String = "./oist_rae_weight.npy"

  @Option(help: "Which bounding box to track")
  var boxId: Int = 0

  @Option(help: "Track for how many frames")
  var trackFrames: Int = 10

  @Option(help: "Track the target from frame x")
  var trackStartFrame: Int = 250

  @Option(help: "The dimension of the latent code in the RAE appearance model")
  var kLatentDimension = 10

  @Flag(help: "Print progress information")
  var verbose: Bool = false

  /// Returns predictions for `videoName` using the raw pixel tracker.
  func naivePpcaTrack(dataset dataset_: OISTBeeVideo, length: Int, startFrom: Int) -> [OrientedBoundingBox] {
    var dataset = dataset_
    dataset.labels = dataset.labels.map {
      $0.filter({ $0.label == .Body })
    }
    // Make batch and do RAE
    let (batch, _) = dataset.makeBatch(appearanceModelSize: (40, 70), batchSize: 300)
    var statistics = FrameStatistics(batch)
    statistics.mean = Tensor(62.26806976644069)
    statistics.standardDeviation = Tensor(37.44683834503672)

    let backgroundBatch = dataset.makeBackgroundBatch(
      patchSize: (40, 70), appearanceModelSize: (40, 70),
      statistics: statistics,
      batchSize: 300
    )

    var ppca = PPCA(latentSize: kLatentDimension)

    ppca.train(images: batch)
    
    if verbose { print("Fitting Naive Bayes model") }

    let batchPositive = ppca.encode(batch)
    let foregroundModel = MultivariateGaussian(from:batchPositive, regularizer: 1e-3)

    let batchNegative = ppca.encode(backgroundBatch)
    let backgroundModel = GaussianNB(from: batchNegative, regularizer: 1e-3)

    if verbose {
      print("Foreground: \(foregroundModel)")
      print("Background: \(backgroundModel)")
    }

    if verbose { print("Loading video frames...") }
    startTimer("VIDEO_LOAD")
    // Load the video and take a slice of it.
    let videos = (0..<length).map { (i) -> Tensor<Float> in
      return withDevice(.cpu) { dataset.loadFrame(dataset.frameIds[startFrom + i])! }
    }
    stopTimer("VIDEO_LOAD")

    let startPose = dataset.labels[startFrom][boxId].location.center

    if verbose {
      print("Creating tracker, startPose = \(startPose)")
    }
    
    startTimer("MAKE_GRAPH")
    var tracker = makeNaiveBayesPCATracker(
      model: ppca,
      statistics: statistics,
      frames: videos,
      targetSize: (dataset.labels[startFrom][boxId].location.rows, dataset.labels[startFrom][boxId].location.cols),
      foregroundModel: foregroundModel, backgroundModel: backgroundModel
    )
    stopTimer("MAKE_GRAPH")

    if verbose { print("Starting Optimization...") }
    // For LM Optimizer
    // if verbose { tracker.optimizer.verbosity = .SUMMARY }

    // tracker.optimizer.cgls_precision = 1e-7
    // tracker.optimizer.precision = 1e-4
    // tracker.optimizer.max_iteration = 200

    startTimer("GRAPH_INFER")
    let prediction = tracker.infer(knownStart: Tuple1(startPose))
    stopTimer("GRAPH_INFER")

    let boxes = tracker.frameVariableIDs.map { frameVariableIDs -> OrientedBoundingBox in
      let poseID = frameVariableIDs.head
      return OrientedBoundingBox(
        center: prediction[poseID], rows: dataset.labels[startFrom][boxId].location.rows, cols: dataset.labels[startFrom][boxId].location.cols)
    }

    return boxes
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

    startTimer("PPCA_TRACKING")
    var bboxes: [OrientedBoundingBox]
    bboxes = naivePpcaTrack(dataset: dataset, length: trackFrames, startFrom: trackStartFrame)
    stopTimer("PPCA_TRACKING")

    let frameRawId = dataset.frameIds[trackStartFrame + trackFrames]
    let image = dataset.loadFrame(frameRawId)!

    if verbose {
      print("Creating output plot")
    }
    startTimer("PLOTTING")
    // plot(image, boxes: bboxes.indices.map {
    //   ("\($0)", bboxes[$0])
    // }, margin: 10.0, scale: 0.5).show()
    stopTimer("PLOTTING")

    if verbose {
      printTimers()
    }
  }
}

struct VisualizeTrack: ParsableCommand {
  @Option(help: "Index of the track to visualize")
  var trackIndex: Int

  @Option(help: "Directory for the output frames")
  var output: String

  func run() {
    let dataset = OISTBeeVideo(deferLoadingFrames: true)!
    dataset.tracks[trackIndex].render(to: output, video: dataset)
  }
}

// It is important to set the global threadpool before doing anything else, so that nothing
// accidentally uses the default threadpool.
ComputeThreadPools.global =
  NonBlockingThreadPool<PosixConcurrencyPlatform>(name: "mypool", threadCount: 12)

OISTVisualizationTool.main()
