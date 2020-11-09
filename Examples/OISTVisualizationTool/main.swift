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
    subcommands: [ViewFrame.self, PpcaTrack.self])
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
func makeOISTBatch(dataset: OISTBeeVideo, appearanceModelSize: (Int, Int))
  -> (normalized: Tensor<Double>, statistics: FrameStatistics)
{
  let images = Array(dataset.frameIds.indices.lazy.map { (frameId: Int) -> [Tensor<Double>] in
    let currentFrame = dataset.loadFrame(dataset.frameIds[frameId])!
    return dataset.labels[frameId].map {
      currentFrame.patch(at: $0.location, outputSize: appearanceModelSize)
    }
  }.joined())

  let stacked = Tensor(stacking: images)
  let statistics = FrameStatistics(stacked)
  return (statistics.normalized(stacked), statistics)
}

/// Tracking with a PPCA graph
struct PpcaTrack: ParsableCommand {
  @Option(help: "Location of dataset folder which should contain `frames` and `frames_txt`")
  var datasetLocation: String = "./OIST_Data"

  @Option(help: "Which bounding box to track")
  var boxId: Int = 0

  @Option(help: "Track for how many frames")
  var trackFrames: Int = 0

  @Flag(help: "Print progress information")
  var verbose: Bool = false


  /// Returns predictions for `videoName` using the raw pixel tracker.
  func rawPixelTrack(dataset: OISTBeeVideo, length: Int) -> [OrientedBoundingBox] {
    // Load the video and take a slice of it.
    let videos = (0..<length).map {
      dataset.loadFrame(dataset.frameIds[$0])!
    }
    
    let startPatch = videos[0].patch(at: dataset.labels[0][boxId].location)
    let startPose = dataset.labels[0][boxId].location.center

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
    let dataset = OISTBeeVideo(directory: dataURL, deferLoadingFrames: true)!

    ComputeThreadPools.global = NonBlockingThreadPool<PosixConcurrencyPlatform>(name: "mypool", threadCount: 12)

    let bboxes = rawPixelTrack(dataset: dataset, length: trackFrames)

    let frameRawId = dataset.frameIds[trackFrames]
    let image = dataset.loadFrame(frameRawId)!
    plot(image, boxes: bboxes.indices.map {
      ("\($0)", bboxes[$0])
    }, margin: 10.0, scale: 0.5).show()
  }
}

OISTVisualizationTool.main()
