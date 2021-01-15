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
  var featureSize: Int = 100

  // Just runs an RP tracker and saves image to file
  // Make sure you have a folder `Results/frank02` before running
  func run() {
    let dataDir = URL(fileURLWithPath: "./OIST_Data")
    let data = OISTBeeVideo(directory: dataDir)!
    
    
    let (imageHeight, imageWidth, imageChannels) =
      (40, 70, 1)
    
    let rp = RandomProjection(fromShape: TensorShape([imageHeight, imageWidth, imageChannels]), toFeatureSize: featureSize)

    let trackerEvaluation = TrackerEvaluationDataset(data)
    
    let evalTracker: Tracker = {frames, start in
        let trainingDatasetSize = 100
        var tracker = trainProbabilisticTracker(
            trainingData: data,
            encoder: rp,
            frames: frames,
            boundingBoxSize: (40, 70),
            withFeatureSize: 100,
            fgRandomFrameCount: trainingDatasetSize,
            bgRandomFrameCount: trainingDatasetSize,
            numberOfTrainingSamples: 3000
        )
        //   trainingData = OISTBeeVideo(directory:  URL(fileURLWithPath: "./OIST_Data"), length: 100)!
        //   var tracker = trainRPTracker(~
        //   trainingData: trainingData,
        //   frames: frames, boundingBoxSize: (40, 70), withFeatureSize: 100, usingEM :false
        // )
        let prediction = tracker.infer(knownStart: Tuple1(start.center), withSampling: false)//).map { OrientedBoundingBox(center: $0, rows: 70, cols: 40)}
        print(type(of: prediction))
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