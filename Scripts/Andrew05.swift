import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation

import PenguinStructures

/// Andrew01: RAE Tracker
struct Andrew05: ParsableCommand {

  @Option(help: "Run for number of frames")
  var trackLength: Int = 80
  
  @Option(help: "Size of feature space")
  var featureSize: Int = 256

  @Option(help: "Pretrained weights")
  var weightsFile: String?


  // Comparison SiamMask and RAE
  func run() {
    let np = Python.import("numpy")
    let plt = Python.import("matplotlib.pyplot")
    let pickle = Python.import("pickle")

    let trainingDatasetSize = 100

    let dataDir = URL(fileURLWithPath: "./OIST_Data")
    let data = OISTBeeVideo(directory: dataDir, length: trainingDatasetSize)!
    let testData = OISTBeeVideo(directory: dataDir, afterIndex: trainingDatasetSize, length: trackLength)!
    let testData2 = OISTBeeVideo(directory: dataDir, afterIndex: trainingDatasetSize, length: trackLength)!

    let trackerEvaluation = TrackerEvaluationDataset(testData)
    
    var i = 0
    let evalTrackerSiam: Tracker = {frames, start in
        let decoder = JSONDecoder()
        let trackPath = "Results/brando03/prediction_siammask_sequence_\(i).json"
        let decodedTrack = try! decoder.decode([OrientedBoundingBox].self, from: Data(contentsOf: URL(fileURLWithPath: trackPath)))
        i = i + 1
        return decodedTrack
    }
    let sequenceCount = 19

    var results_siam = trackerEvaluation.evaluate(evalTrackerSiam, sequenceCount: sequenceCount, deltaAnchor: 175, outputFile: "andrew02")
    i = 0
    let trackerEvaluation2 = TrackerEvaluationDataset(testData2)
    let evalTrackerRae: Tracker = {frames, start in
        let decoder = JSONDecoder()
        let trackPath = "rae_256_updated_preds/prediction_rae_256_sequence_\(i).json"
        let decodedTrack = try! decoder.decode([OrientedBoundingBox].self, from: Data(contentsOf: URL(fileURLWithPath: trackPath)))
        i = i + 1
        return decodedTrack
    }
    var results_rae = trackerEvaluation2.evaluate(evalTrackerRae, sequenceCount: sequenceCount, deltaAnchor: 175, outputFile: "andrew01")

    
    

    // for (index, value) in results_rae.sequences.prefix(sequenceCount).enumerated() {
    for j in 0...sequenceCount-1 {
      let value_rae = results_rae.sequences.prefix(sequenceCount)[j]
      let index = j
      let value_siam =  results_siam.sequences.prefix(sequenceCount)[j]
      let value = value_rae
      var i: Int = 0
      // zip(value.subsequences.first!.frames, zip(value.subsequences.first!.prediction, value.subsequences.first!.groundTruth)).map {
      //   let (fig, axes) = plotFrameWithPatches(frame: $0.0, actual: $0.1.0.center, expected: $0.1.1.center, firstGroundTruth: value.subsequences.first!.groundTruth.first!.center)
      //   fig.savefig("Results/brando03/sequence\(index)/brando03\(i).png", bbox_inches: "tight")
      //   plt.close("all")
      //   i = i + 1
      // }
      zip(value_rae.subsequences.first!.frames, zip(zip(value_rae.subsequences.first!.prediction,value_siam.subsequences.first!.prediction), value_rae.subsequences.first!.groundTruth)).map {
        // let (fig, axes) = plotFrameWithPatches(frame: $0.0, actual: $0.1.0.center, expected: $0.1.1.center, firstGroundTruth: value.subsequences.first!.groundTruth.first!.center)
        let (fig, _) = plotFrameWithPatches2(frame: $0.0, actual_box1: $0.1.0.0, actual_box2: $0.1.0.1, expected: $0.1.1.center, firstGroundTruth: value_rae.subsequences.first!.groundTruth.first!.center)
        fig.savefig("Results/andrew01/sequence\(index)/comparison_\(i).png", bbox_inches: "tight")
        plt.close("all")
        i = i + 1
      }
      
      // let (fig, axes) = plt.subplots(1, 2, figsize: Python.tuple([20, 20])).tuple2
      // fig.suptitle("Tracking positions and Subsequence Average Overlap with Accuracy \(String(format: "%.2f", value_rae.subsequences.first!.metrics.accuracy)) and Robustness \(value_rae.subsequences.first!.metrics.robustness).")
      // print("First Ground Truth")
      // value_rae.subsequences.map {
      //   print($0.prediction.first!)
      //   $0.prediction.map{print("\(round($0.center.t.x)) \(round($0.center.t.y)) \($0.center.rot.theta) \(40) \(70)")}
        
      //   plotPoseDifference(
      //     track: $0.prediction.map{$0.center}, withGroundTruth: $0.groundTruth.map{$0.center}, on: axes[0]
      //   )
      // }
      // plotOverlap(
      //     metrics: value_rae.subsequences.first!.metrics, on: axes[1]
      // )
      // fig.savefig("Results/andrew01/andrew01_subsequence\(index).png", bbox_inches: "tight")
      print("Accuracy for sequence is \(value_rae.sequenceMetrics.accuracy) with Robustness of \(value_rae.sequenceMetrics.robustness)")
    }

    // print("Accuracy for all sequences is \(results.trackerMetrics.accuracy) with Robustness of \(results.trackerMetrics.robustness)")
    // let f = Python.open("Results/EAO/rae_em_\(featureSize).data", "wb")
    // pickle.dump(results.expectedAverageOverlap.curve, f)


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