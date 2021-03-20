import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation

import PenguinStructures

/// Fan10: RP Tracker, using the new tracking model
struct Andrew04: ParsableCommand {
  
  typealias LikelihoodModel = TrackingLikelihoodModel<RandomProjection, MultivariateGaussian, MultivariateGaussian>

  @Option(help: "Run on track number x")
  var trackId: Int = 3

  @Option(help: "Run for number of frames")
  var trackLength: Int = 80

  @Option(help: "Size of feature space")
  var featureSize: Int = 256

  @Flag(help: "Training mode")
  var training: Bool = false

  func getTrainingDataEM(
    from dataset: OISTBeeVideo,
    numberForeground: Int = 3000,
    numberBackground: Int = 3000
  ) -> [LikelihoodModel.Datum] {
    let bgBoxes = dataset.makeBackgroundBoundingBoxes(patchSize: (40, 70), batchSize: numberBackground).map {
      (frame: $0.frame, type: LikelihoodModel.PatchType.bg, obb: $0.obb)
    }
    let fgBoxes = dataset.makeForegroundBoundingBoxes(patchSize: (40, 70), batchSize: numberForeground).map {
      (frame: $0.frame, type: LikelihoodModel.PatchType.fg, obb: $0.obb)
    }//.filter{$0.obb.center.rot.theta != -1.5707963267948966}
    
    return fgBoxes + bgBoxes
  }
  
  // Just runs an RP tracker and saves image to file
  // Make sure you have a folder `Results/fan10` before running
  func run() {
    let np = Python.import("numpy")
    let pickle = Python.import("pickle")

    let plt = Python.import("matplotlib.pyplot")
    let kHiddenDimension = 512
    let trainingDatasetSize = 100
    let dataDir = URL(fileURLWithPath: "./OIST_Data")
    let generator = ARC4RandomNumberGenerator(seed: 42)
    
    var em = MonteCarloEM<LikelihoodModel>(sourceOfEntropy: generator)
    
    //let trainingDataset = OISTBeeVideo(directory: dataDir, length: 30)!
    let data = OISTBeeVideo(directory: dataDir, length: trainingDatasetSize)!
    //let trainingData = getTrainingDataEM(from: data)
    
    var statistics = FrameStatistics(Tensor<Double>(0.0))
    statistics.mean = Tensor(62.26806976644069)
    statistics.standardDeviation = Tensor(37.44683834503672)
    let likelihoodModel = em.run(
      with: getTrainingDataEM(from: data),
      iterationCount: 3,
      hook: { i, _, _ in
        print("EM run iteration \(i)")
      },
      given: LikelihoodModel.HyperParameters(
        encoder: PretrainedDenseRAE.HyperParameters(hiddenDimension: kHiddenDimension, latentDimension: featureSize, weightFile: "./oist_rae_weight_\(featureSize).npy"), frameStatistics: statistics
      )
    )
    print("at test data!")
    let testData = OISTBeeVideo(directory: dataDir, afterIndex: trainingDatasetSize, length: trackLength)!

    
    // var statistics = FrameStatistics(Tensor<Double>(0.0))
    // statistics.mean = Tensor(62.26806976644069)
    // statistics.standardDeviation = Tensor(37.44683834503672)
    // let trainingBatch = data.makeBatch(statistics: statistics, appearanceModelSize: (imageHeight, imageWidth), batchSize: 3000)
    // let rp = PCAEncoder(from: trainingBatch, given: featureSize)
    let trackerEvaluation = TrackerEvaluationDataset(testData)
    print("created dataset")
    var i = 0
    let evalTracker: Tracker = {frames, start in
        var tracker = makeProbabilisticTracker(
            model: likelihoodModel.encoder,
            frames: frames, targetSize: (40, 70),
            foregroundModel: likelihoodModel.foregroundModel, backgroundModel: likelihoodModel.backgroundModel
        )
        
        let prediction = tracker.infer(knownStart: Tuple1(start.center), withSampling: true)
        let track = tracker.frameVariableIDs.map { OrientedBoundingBox(center: prediction[unpack($0)], rows: 40, cols:70) }
        
        i = i + 1
        return track
    }
    
    let sequenceCount = 1
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