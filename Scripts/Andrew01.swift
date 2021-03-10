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
  @Option(help: "Run for number of frames")
  var trackLength: Int = 80
  
  @Option(help: "Size of feature space")
  var featureSize: Int = 256

  @Option(help: "Pretrained weights")
  var weightsFile: String?

  // Runs RAE tracker on n number of sequences and outputs relevant images and statistics
  // Make sure you have a folder `Results/andrew01` before running
  func run() {
    let np = Python.import("numpy")
    let plt = Python.import("matplotlib.pyplot")
    let pickle = Python.import("pickle")
    let kHiddenDimension = 512

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
    //let rp = RandomProjection(fromShape: TensorShape([imageHeight, imageWidth, imageChannels]), toFeatureSize: featureSize)

    let trainingDatasetSize = 100

    let dataDir = URL(fileURLWithPath: "./OIST_Data")
    let data = OISTBeeVideo(directory: dataDir, length: trainingDatasetSize)!
    let testData = OISTBeeVideo(directory: dataDir, afterIndex: trainingDatasetSize, length: trackLength)!

    // var statistics = FrameStatistics(Tensor<Double>(0.0))
    // statistics.mean = Tensor(62.26806976644069)
    // statistics.standardDeviation = Tensor(37.44683834503672)
    // let trainingBatch = data.makeBatch(statistics: statistics, appearanceModelSize: (imageHeight, imageWidth), batchSize: 3000)
    // let rp = PCAEncoder(from: trainingBatch, given: featureSize)
    let trackerEvaluation = TrackerEvaluationDataset(testData)
    var i = 0
    let evalTracker: Tracker = {frames, start in
        var tracker = trainProbabilisticTracker(
            trainingData: data,
            encoder: rae,
            frames: frames,
            boundingBoxSize: (40, 70),
            withFeatureSize: featureSize,
            fgRandomFrameCount: trainingDatasetSize,
            bgRandomFrameCount: trainingDatasetSize,
            numberOfTrainingSamples: 3000
        )
        
        let prediction = tracker.infer(knownStart: Tuple1(start.center), withSampling: true)
        let track = tracker.frameVariableIDs.map { OrientedBoundingBox(center: prediction[unpack($0)], rows: 40, cols:70) }
        
        i = i + 1
        return track
    }
    
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
        //zip($0.prediction, $0.groundTruth).enumerated().map{($0.0, $0.1.0.center, $0.1.1.center)})
        let encoder = JSONEncoder()
        let data = try! encoder.encode($0.prediction)
        FileManager.default.createFile(atPath: "prediction_rae_\(featureSize)_sequence_\(index).json", contents: data, attributes: nil)
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
    // let f = Python.open("Results/EAO/rp_\(featureSize).data", "wb")
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