import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation

import PenguinStructures

/// Andrew01: RAE Tracker
struct Andrew01: ParsableCommand {
  @Option(help: "Run on track number x")
  var trackId: Int = 0
  
  @Option(help: "Run for number of frames")
  var trackLength: Int = 80
  
  @Option(help: "Size of feature space")
  var featureSize: Int = 5

  @Option(help: "Pretrained weights")
  var weightsFile: String?

  // Runs RAE tracker on n number of sequences and outputs relevant images and statistics
  // Make sure you have a folder `Results/andrew01` before running
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
            withFeatureSize: 5,
            fgRandomFrameCount: trainingDatasetSize,
            bgRandomFrameCount: trainingDatasetSize
        )
        let prediction = tracker.infer(knownStart: Tuple1(start.center), withSampling: true)
        let track = tracker.frameVariableIDs.map { OrientedBoundingBox(center: prediction[unpack($0)], rows: 40, cols:70) }

        return track
    }

    var results = trackerEvaluation.evaluate(evalTracker, sequenceCount: 5, deltaAnchor: 175, outputFile: "andrew01")
    
    
    for (index, value) in results.sequences.prefix(5).enumerated() {
      var i: Int = 0
      zip(value.subsequences.first!.frames, zip(value.subsequences.first!.prediction, value.subsequences.first!.groundTruth)).map {
        let (fig, axes) = plotPatchWithGT(frame: $0.0, actual: $0.1.0.center, expected: $0.1.1.center)
        fig.savefig("Results/andrew01/sequence\(index)/andrew01_\(i).png", bbox_inches: "tight")
        i = i + 1
      }
      let plt = Python.import("matplotlib.pyplot")
      let (fig, axes) = plt.subplots(1, 2, figsize: Python.tuple([20, 20])).tuple2
      fig.suptitle("Tracking positions and Subsequence Average Overlap with Accuracy \(String(format: "%.2f", value.subsequences.first!.metrics.accuracy)) and Robustness \(value.subsequences.first!.metrics.robustness).")
      
      value.subsequences.map {
        plotTrajectory(
          track: $0.prediction.map{$0.center}, withGroundTruth: $0.groundTruth.map{$0.center}, on: axes[0],
          withTrackColors: plt.cm.jet, withGtColors: plt.cm.gray
        )
      }
      plotOverlap(
          metrics: value.subsequences.first!.metrics, on: axes[1]
      )
      fig.savefig("Results/andrew01/andrew01_subsequence\(index).pdf", bbox_inches: "tight")
    }
    


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