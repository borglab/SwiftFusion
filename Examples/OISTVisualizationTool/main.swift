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
    subcommands: [VisualizeTrack.self, ViewFrame.self, RawTrack.self, PpcaTrack.self])
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

    plot(image, boxes: dataset.labels[frameId].enumerated().map {
      (String($0), $1.location)
    }, margin: 10.0, scale: 0.5).show()
  }
}

/// Returns a `[N, h, w, c]` batch of normalized patches from a VOT video, and returns the
/// statistics used to normalize them.
func makeOISTBatch(dataset: OISTBeeVideo, appearanceModelSize: (Int, Int), batchSize: Int = 200)
  -> (normalized: Tensor<Double>, statistics: FrameStatistics)
{
  var images: [Tensor<Double>] = []
  images.reserveCapacity(batchSize)

  var currentFrame: Tensor<Double> = [0]
  var currentId: Int = -1

  var deterministicEntropy = ARC4RandomNumberGenerator(seed: 42)
  for label in dataset.labels.randomSelectionWithoutReplacement(k: 10, using: &deterministicEntropy).lazy.joined().randomSelectionWithoutReplacement(k: batchSize, using: &deterministicEntropy).sorted(by: { $0.frameIndex < $1.frameIndex }) {
    if currentId != label.frameIndex {
      currentFrame = dataset.loadFrame(label.frameIndex)!
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
    let videos = (0..<length).map { (i) -> Tensor<Double> in
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

    if verbose { tracker.optimizer.verbosity = .SUMMARY }

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

    ComputeThreadPools.global = NonBlockingThreadPool<PosixConcurrencyPlatform>(name: "mypool", threadCount: 12)

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
    plot(image, boxes: bboxes.indices.map {
      ("\($0)", bboxes[$0])
    }, margin: 10.0, scale: 0.5).show()
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
    let videos = (0..<length).map { (i) -> Tensor<Double> in
      return withDevice(.cpu) { dataset.loadFrame(dataset.frameIds[i])! }
    }
    stopTimer("VIDEO_LOAD")

    let startPatch = statistics.normalized(videos[0].patch(at: dataset.labels[0][boxId].location))
    let startPose = dataset.labels[0][boxId].location.center
    let startLatent = ppca.encode(startPatch)

    if verbose {
      print("Creating tracker, startPose = \(startPose)")
    }
    
    startTimer("MAKE_GRAPH")
    var tracker = makePPCATracker(model: ppca, statistics: statistics, frames: videos, targetSize: (40, 70))
    stopTimer("MAKE_GRAPH")

    if verbose { tracker.optimizer.verbosity = .SUMMARY }

    tracker.optimizer.cgls_precision = 1e-6
    tracker.optimizer.precision = 1e-2

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

    ComputeThreadPools.global = NonBlockingThreadPool<PosixConcurrencyPlatform>(name: "mypool", threadCount: 12)

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
    plot(image, boxes: bboxes.indices.map {
      ("\($0)", bboxes[$0])
    }, margin: 10.0, scale: 0.5).show()
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

OISTVisualizationTool.main()
