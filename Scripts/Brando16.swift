import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation

import PenguinStructures

/// PCA tests
struct Brando16: ParsableCommand {
  typealias LikelihoodModel = TrackingLikelihoodModel<PretrainedDenseRAE, MultivariateGaussian, GaussianNB>

  @Option(help: "Run for number of frames")
  var trackLength: Int = 80


   func getTrainingData(
    from dataset: OISTBeeVideo,
    numberForeground: Int = 4500
  ) -> [LikelihoodModel.Datum] {
    let fgBoxes = dataset.makeForegroundBoundingBoxes(patchSize: (40, 70), batchSize: numberForeground).map {
      (frame: $0.frame, type: LikelihoodModel.PatchType.fg, obb: $0.obb)
    }
    
    return fgBoxes
  }

  // Runs RAE tracker on n number of sequences and outputs relevant images and statistics
  // Make sure you have a folder `Results/andrew01` before running
  func run() {
    let np = Python.import("numpy")
    let pickle = Python.import("pickle")
    // used to be 512

    let (imageHeight, imageWidth, imageChannels) =
      (40, 70, 1)


    var kHiddenDimension = [16, 64, 256]   
    for dim in kHiddenDimension {
        let dataDir = URL(fileURLWithPath: "./OIST_Data")

        let trainingDataset = OISTBeeVideo(directory: dataDir, length: 100)!
        
        let trainingData = Tensor<Double>(stacking: getTrainingData(from: trainingDataset).map { $0.frame!.patch(at: $0.obb) })
        let testData = OISTBeeVideo(directory: dataDir, afterIndex: 100, length: trackLength)!


        var statistics = FrameStatistics(Tensor<Double>(0.0))
        statistics.mean = Tensor(62.26806976644069)
        statistics.standardDeviation = Tensor(37.44683834503672)
        let trainingBatch = trainingDataset.makeBatch(statistics: statistics, appearanceModelSize: (imageHeight, imageWidth), batchSize: 4500)
        let rae = PCAEncoder(from: trainingBatch, given: dim)
        


        let trackerEvaluation = TrackerEvaluationDataset(testData)
        print("s1")
        let evalTracker: Tracker = {frames, start in
            var tracker = trainProbabilisticTracker(
                trainingData: trainingDataset,
                encoder: rae,
                frames: frames,
                boundingBoxSize: (40, 70),
                withFeatureSize: dim,
                fgRandomFrameCount: 100,
                bgRandomFrameCount: 100
            )
            let prediction = tracker.infer(knownStart: Tuple1(start.center), withSampling: true)
            let track = tracker.frameVariableIDs.map { OrientedBoundingBox(center: prediction[unpack($0)], rows: 40, cols:70) }

            return track
        }
        let plt = Python.import("matplotlib.pyplot")
        let sequenceCount = 19
        var results = trackerEvaluation.evaluate(evalTracker, sequenceCount: sequenceCount, deltaAnchor: 175, outputFile: "andrew01")

              for (index, value) in results.sequences.prefix(sequenceCount).enumerated() {
                var i: Int = 0
                zip(value.subsequences.first!.frames, zip(value.subsequences.first!.prediction, value.subsequences.first!.groundTruth)).map {
                  let (fig, axes) = plotFrameWithPatches(frame: $0.0, actual: $0.1.0.center, expected: $0.1.1.center, firstGroundTruth: value.subsequences.first!.groundTruth.first!.center)
                  fig.savefig("Results/ppca_\(dim)/sequence\(index)/andrew01_\(i).png", bbox_inches: "tight")
                  plt.close("all")
                  i = i + 1
                }
                
                
                let (fig, axes) = plt.subplots(1, 2, figsize: Python.tuple([20, 20])).tuple2
                fig.suptitle("Tracking positions and Subsequence Average Overlap with Accuracy \(String(format: "%.2f", value.subsequences.first!.metrics.accuracy)) and Robustness \(value.subsequences.first!.metrics.robustness).")
                
                value.subsequences.map {
                  let encoder = JSONEncoder()
                  let data = try! encoder.encode($0.prediction)
                  FileManager.default.createFile(atPath: "Results/ppca_\(dim)/prediction_ppca_\(dim)_sequence_\(index).json", contents: data, attributes: nil)
                  plotPoseDifference(
                    track: $0.prediction.map{$0.center}, withGroundTruth: $0.groundTruth.map{$0.center}, on: axes[0]
                  )
                }
                plotOverlap(
                    metrics: value.subsequences.first!.metrics, on: axes[1]
                )
                fig.savefig("Results/ppca_\(dim)/andrew01_subsequence\(index).png", bbox_inches: "tight")
                print("Accuracy for sequence is \(value.sequenceMetrics.accuracy) with Robustness of \(value.sequenceMetrics.robustness)")
              }

              print("Accuracy for all sequences is \(results.trackerMetrics.accuracy) with Robustness of \(results.trackerMetrics.robustness)")
              let f = Python.open("Results/ppca_\(dim)/EAO/rp_\(dim).data", "wb")
              pickle.dump(results.expectedAverageOverlap.curve, f)


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