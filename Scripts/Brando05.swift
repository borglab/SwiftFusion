import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation

import PenguinStructures

/// Brando05: TRACKING with NN Classifier
struct Brando05: ParsableCommand {
  @Option(help: "Run for number of frames")
  var trackLength: Int = 80

  // Runs NNClassifier tracker on n number of sequences and outputs relevant images and statistics
  func run() {
    let np = Python.import("numpy")
    let featureSizes = [256]
    let kHiddenDimensions = [512]
    let iterations = [1,2,3,4,5,6,7]
    let trainingDatasetSize = 100

    let dataDir = URL(fileURLWithPath: "./OIST_Data")
    let data = OISTBeeVideo(directory: dataDir, length: trainingDatasetSize)!
    let testData = OISTBeeVideo(directory: dataDir, afterIndex: trainingDatasetSize, length: trackLength)!

    let trackerEvaluation = TrackerEvaluationDataset(testData)

    for featureSize in featureSizes {
    for kHiddenDimension in kHiddenDimensions {
    for j in iterations {


    let (imageHeight, imageWidth, imageChannels) =
      (40, 70, 1)

    // var classifier = SmallerNNClassifier(
    //   imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels, latentDimension: featureSize
    // )
    var classifier = LargerNNClassifier(
      imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels, hiddenDimension: kHiddenDimension, latentDimension: featureSize
    )
    // LOAD THE CLASSIFIER
    // classifier.load(weights: np.load("./classifiers/classifiers_today/classifier_weight_\(kHiddenDimension)_\(featureSize)_\(j)_doubletraining.npy", allow_pickle: true))
    classifier.load(weights: np.load("./classifiers/classifiers_today/large_classifier_weight_\(kHiddenDimension)_\(featureSize)_\(j).npy", allow_pickle: true))
    // classifier.load(weights: np.load("./classifiers/classifiers_today/small_classifier_weight_\(featureSize)_\(j).npy", allow_pickle: true))

    let evalTracker: Tracker = {frames, start in
        var tracker = makeProbabilisticTracker2(
            model: classifier,
            frames: frames,
            targetSize: (40, 70)
        )
        let prediction = tracker.infer(knownStart: Tuple1(start.center), withSampling: true)
        let track = tracker.frameVariableIDs.map { OrientedBoundingBox(center: prediction[unpack($0)], rows: 40, cols:70) }
        return track

    }
    // print(evalTracker)
    // return

    let plt = Python.import("matplotlib.pyplot")
    let sequenceCount = 1
    var results = trackerEvaluation.evaluate(evalTracker, sequenceCount: sequenceCount, deltaAnchor: 175, outputFile: "classifier")


    for (index, value) in results.sequences.prefix(sequenceCount).enumerated() {
      let folderName = "Results/classifier/classifier_\(kHiddenDimension)_\(featureSize)_\(j)_10000sampling"
      print(folderName)
      if !FileManager.default.fileExists(atPath: folderName) {
      do {
          try FileManager.default.createDirectory(atPath: folderName, withIntermediateDirectories: true, attributes: nil)
          // print("here")
          try FileManager.default.createDirectory(atPath: folderName + "/sequence0", withIntermediateDirectories: true, attributes: nil)
          // print("here2")
      } catch {
          print(error.localizedDescription)
      }
      }

      var i: Int = 0
      zip(value.subsequences.first!.frames, zip(value.subsequences.first!.prediction, value.subsequences.first!.groundTruth)).map {
        let (fig, axes) = plotFrameWithPatches(frame: $0.0, actual: $0.1.0.center, expected: $0.1.1.center, firstGroundTruth: value.subsequences.first!.groundTruth.first!.center)
        fig.savefig(folderName + "/sequence\(index)/classifier_\(i).png", bbox_inches: "tight")
        plt.close("all")
        i = i + 1
      }
      
      let (fig, axes) = plt.subplots(1, 2, figsize: Python.tuple([20, 20])).tuple2
      fig.suptitle("Tracking positions and Subsequence Average Overlap with Accuracy \(String(format: "%.2f", value.subsequences.first!.metrics.accuracy)) and Robustness \(value.subsequences.first!.metrics.robustness).")
      
      value.subsequences.map {
        plotPoseDifference(
          track: $0.prediction.map{$0.center}, withGroundTruth: $0.groundTruth.map{$0.center}, on: axes[0]
        )
      }
      plotOverlap(
          metrics: value.subsequences.first!.metrics, on: axes[1]
      )
      
      fig.savefig(folderName + "/classifier_\(kHiddenDimension)_\(featureSize)_\(j)subsequence\(index).png", bbox_inches: "tight")
      print("Accuracy for sequence is \(value.sequenceMetrics.accuracy) with Robustness of \(value.sequenceMetrics.robustness)")
    }

    print("Accuracy for all sequences is \(results.trackerMetrics.accuracy) with Robustness of \(results.trackerMetrics.robustness)")
    



    }
    }
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