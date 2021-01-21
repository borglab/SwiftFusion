import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation

import PenguinStructures

/// Fan03: RP Tracker, with sampling-based initialization
struct Andrew01: ParsableCommand {
  @Option(help: "Run on track number x")
  var trackId: Int = 0
  
  @Option(help: "Run for number of frames")
  var trackLength: Int = 80
  
  @Option(help: "Size of feature space")
  var featureSize: Int = 5

  @Option(help: "Pretrained weights")
  var weightsFile: String?

  // Just runs an RP tracker and saves image to file
  // Make sure you have a folder `Results/frank02` before running
  func run() {
    let np = Python.import("numpy")
    let kHiddenDimension = 100

    let (imageHeight, imageWidth, imageChannels) =
      (40, 70, 1)
    var rae = DenseRAE(
      imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels,
      hiddenDimension: kHiddenDimension, latentDimension: featureSize
    )

    if let weightsFile = weightsFile {
      rae.load(weights: np.load(weightsFile, allow_pickle: true))
    } else {
      rae.load(weights: np.load("./oist_rae_weight_\(featureSize).npy", allow_pickle: true))
    }

    let dataDir = URL(fileURLWithPath: "./OIST_Data")
    let data = OISTBeeVideo(directory: dataDir, length: 100)!
    let testData = OISTBeeVideo(directory: dataDir, afterIndex: 100, length: 80)!

    let trackerEvaluation = TrackerEvaluationDataset(testData)
    
    let evalTracker: Tracker = {frames, start in
        let trainingDatasetSize = 100
        var tracker = trainProbabilisticTracker(
            trainingData: data,
            encoder: rae,
            frames: frames,
            boundingBoxSize: (40, 70),
            withFeatureSize: 100,
            fgRandomFrameCount: trainingDatasetSize,
            bgRandomFrameCount: trainingDatasetSize,
            numberOfTrainingSamples: 3000
        )
        let prediction = tracker.infer(knownStart: Tuple1(start.center), withSampling: false)
        let track = tracker.frameVariableIDs.map { OrientedBoundingBox(center: prediction[unpack($0)], rows: 40, cols:70) }
        return track
    }
  
    trackerEvaluation.evaluate(evalTracker, sequenceCount: 3, deltaAnchor: 360, outputFile: "andrew01")

  }
}

/// Returns `t` as a Swift tuple.
fileprivate func unpack<A, B>(_ t: Tuple2<A, B>) -> (A, B) {
  return (t.head, t.tail.head)
}
/// Returns `t` as a Swift tuple.
fileprivate func unpack<A>(_ t: Tuple1<A>) -> (A) {
  return (t.head)
}