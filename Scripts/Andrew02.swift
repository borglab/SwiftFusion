import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation

import PenguinStructures

/// Andrew02: RAE Tracker with expectation maximization
struct Andrew02: ParsableCommand {
  @Option(help: "Run for number of frames")
  var trackLength: Int = 80
  
  @Option(help: "Size of feature space")
  var featureSize: Int = 256

  @Option(help: "Pretrained weights")
  var weightsFile: String?

  typealias CurrentModel = ProbablisticTracker<PretrainedDenseRAE, MultivariateGaussian, MultivariateGaussian>
  func getTrainingDataEM(
  from dataset: OISTBeeVideo,
  allBgBoxes: [(frameID: Int, frame: Tensor<Double>?, obb: OrientedBoundingBox)],
  allFgBoxes: [(frameID: Int, frame: Tensor<Double>?, obb: OrientedBoundingBox)],
  numberForeground: Int = 1000,
  numberBackground: Int = 1000
  ) -> [CurrentModel.Datum] {
    
    let bgBoxes = allBgBoxes.map {
      (frameID: $0.frameID, type: CurrentModel.PatchType.bg, obb: $0.obb)
    }
    let fgBoxes = allFgBoxes.map {
      (frameID: $0.frameID, type: CurrentModel.PatchType.fg, obb: $0.obb)
    }
    
    return fgBoxes + bgBoxes
  }
  // Runs RAE tracker on n number of sequences and outputs relevant images and statistics
  func run() {
    let np = Python.import("numpy")
    let pickle = Python.import("pickle")
    let kHiddenDimension = 512

    let (imageHeight, imageWidth, imageChannels) =
      (40, 70, 1)

    var statistics = FrameStatistics(Tensor<Double>(0.0))
    statistics.mean = Tensor(62.26806976644069)
    statistics.standardDeviation = Tensor(37.44683834503672)

    let trainingDatasetSize = 100

    let dataDir = URL(fileURLWithPath: "./OIST_Data")
    let data = OISTBeeVideo(directory: dataDir, length: trainingDatasetSize)!

    let trainingData = getTrainingDataEM(from: data, allBgBoxes: data.makeBackgroundBoundingBoxes(patchSize: (40, 70), batchSize: 300), allFgBoxes: data.makeForegroundBoundingBoxes(patchSize: (40, 70), batchSize: 300))
    print("I'm here")

    let generator = ARC4RandomNumberGenerator(seed: 42)
    var em = MonteCarloEM<CurrentModel>(sourceOfEntropy: generator)

    let testData = OISTBeeVideo(directory: dataDir, afterIndex: trainingDatasetSize, length: trackLength)!
    let trackerEvaluation = TrackerEvaluationDataset(testData)
    print("now I'm here")
    print(data.frameIds)
    var model = em.run(
            with: trainingData,
            iterationCount: 1,
            sampleCount: 1,
            hook: { i, _, _ in
                print("EM run iteration \(i)")
            },
            given: CurrentModel.HyperParameters(
                encoder: PretrainedDenseRAE.HyperParameters(hiddenDimension: kHiddenDimension, latentDimension: featureSize, weightFile: "./oist_rae_weight_\(featureSize).npy"),
                onFrames: data,
                frameStatistics: statistics
            )
        )
    let evalTracker: Tracker = {frames, start in

        return model.infer(start: start, frames: frames)
    }
    let plt = Python.import("matplotlib.pyplot")
    let sequenceCount = 19
    var results = trackerEvaluation.evaluate(evalTracker, sequenceCount: sequenceCount, deltaAnchor: 175, outputFile: "andrew01")

    for (index, value) in results.sequences.prefix(sequenceCount).enumerated() {
      var i: Int = 0
      zip(value.subsequences.first!.frames, zip(value.subsequences.first!.prediction, value.subsequences.first!.groundTruth)).map {
        let (fig, axes) = plotFrameWithPatches(frame: $0.0, actual: $0.1.0.center, expected: $0.1.1.center, firstGroundTruth: value.subsequences.first!.groundTruth.first!.center)
        fig.savefig("Results/andrew01/sequence\(index)/andrew01_\(i).png", bbox_inches: "tight")
        plt.close("all")
        i = i + 1
      }
      
      
      let (fig, axes) = plt.subplots(1, 2, figsize: Python.tuple([20, 20])).tuple2
      fig.suptitle("Tracking positions and Subsequence Average Overlap with Accuracy \(String(format: "%.2f", value.subsequences.first!.metrics.accuracy)) and Robustness \(value.subsequences.first!.metrics.robustness).")
      
      value.subsequences.map {
        let encoder = JSONEncoder()
        let data = try! encoder.encode($0.prediction)
        FileManager.default.createFile(atPath: "prediction_rae_em_\(featureSize)_sequence_\(index).json", contents: data, attributes: nil)
        plotPoseDifference(
          track: $0.prediction.map{$0.center}, withGroundTruth: $0.groundTruth.map{$0.center}, on: axes[0]
        )
      }
      plotOverlap(
          metrics: value.subsequences.first!.metrics, on: axes[1]
      )
      fig.savefig("Results/andrew01/andrew01_subsequence\(index).png", bbox_inches: "tight")
      print("Accuracy for sequence is \(value.sequenceMetrics.accuracy) with Robustness of \(value.sequenceMetrics.robustness)")
    }

    print("Accuracy for all sequences is \(results.trackerMetrics.accuracy) with Robustness of \(results.trackerMetrics.robustness)")
    let f = Python.open("Results/EAO/rae_\(featureSize)_em.data", "wb")
    pickle.dump(results.expectedAverageOverlap.curve, f)


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