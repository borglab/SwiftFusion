import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation

import PenguinStructures

/// Brando01 Tracker OpenCV
struct Brando01: ParsableCommand {
  // @Option(help: "Run on track number x")
  // var trackId: Int = 0
  
  // @Option(help: "Run for number of frames")
  // var trackLength: Int = 80
  
  // @Option(help: "Size of feature space")
  // var featureSize: Int = 5

  // @Option(help: "Pretrained weights")
  // var weightsFile: String?

  // Runs RAE tracker on n number of sequences and outputs relevant images and statistics
  // Make sure you have a folder `Results/andrew01` before running
  func run() {

    let dataDir = URL(fileURLWithPath: "./OIST_Data")
    let data = OISTBeeVideo(directory: dataDir, length: 100)!
    let testData = OISTBeeVideo(directory: dataDir, afterIndex: 100, length: 80)!
    print("number of frames in training data:", data.labels.count)
    print("number of frames in testing data", testData.labels.count, "\n\n")


    let trackerEvaluation = TrackerEvaluationDataset(testData)

    let np = Python.import("numpy")
    let cv2 = Python.import("cv2")
    // print(Python.version)
    // print("hello")

    let evalTracker: Tracker = {frames, start in
    
      let tracker = cv2.TrackerMIL_create()
      // var tracker = cv2.Tracker_create("MIL")
      // print(frames.first!.makeNumpyArray())
      // BB = (width-35,height-35,70,70)
      //leads to an error when BBox area is more than 40*70?
      var BB = Python.tuple([Int(start.center.t.x)-20, Int(start.center.t.y)-35, 40, 70])
      var smallframe = np.array(frames.first!.makeNumpyArray())
      print("hello2")
      // cv2.circle(smallframe, Python.tuple([Int(start.center.t.x),Int(start.center.t.y)]), 10, Python.tuple([255,255,255]), 5)
      let leftpt = Python.tuple([Int(start.center.t.x)-35, Int(start.center.t.y)-35])
      let rgtpt = Python.tuple([Int(start.center.t.x)+35, Int(start.center.t.y)+35])
      cv2.rectangle(smallframe, leftpt, rgtpt, Python.tuple([0,150,0]), 5)
      print("hello3")
      cv2.imwrite("./image_new.png", smallframe)
      // tracker.init(frames.first!.makeNumpyArray(), BB)
      tracker[dynamicMember: "init"](frames.first!.makeNumpyArray(), BB)
      var results = [PythonObject]()
      for (index, frame) in frames.enumerated() {
        var a = tracker[dynamicMember: "update"](frame.makeNumpyArray()).tuple2
            let track_success = a.0
            let newBB = a.1
            if Bool(track_success)! {
                results.append(newBB)
            }
        // newBB
        // let smallframe = frame.makeNumpyArray()
        // cv2.rectangle(smallframe, leftpt, rgtpt, Python.tuple([0,150,0]), 5)
        // cv2.imshow("SiamMask", smallframe)

               
      }
      print("printing python BB")
      results.map{print($0)}
      // print("hello")
      // print(type(of: results))
      // print(results)
      var track = [OrientedBoundingBox]()
      for result in results {
        let pythonBB = result.tuple4
        let rows = Int(pythonBB.2)!
        let cols = Int(pythonBB.3)!
        let rot = Rot2(0)
        let vect = Vector2(Double(pythonBB.0)!+20, Double(pythonBB.1)!+35)
        // let vect = Vector2(Double(pythonBB.0)! + Double(rows)/2, Double(pythonBB.1)! + Double(cols)/2)
        let center = Pose2(rot, vect)
        let swiftBB = OrientedBoundingBox(center: center, rows: rows, cols: cols)
        track.append(swiftBB)
      }
      return track
    }

  



    let plt = Python.import("matplotlib.pyplot")
    let sequenceCount = 1
    var results = trackerEvaluation.evaluate(evalTracker, sequenceCount: 1, deltaAnchor: 175, outputFile: "brando01")
    // print(results)
    for (index, value) in results.sequences.prefix(1).enumerated() {
      var i: Int = 0
      zip(value.subsequences.first!.frames, zip(value.subsequences.first!.prediction, value.subsequences.first!.groundTruth)).map {
        let (fig, axes) = plotFrameWithPatches(frame: $0.0, actual: $0.1.0.center, expected: $0.1.1.center, firstGroundTruth: value.subsequences.first!.groundTruth.first!.center)
        fig.savefig("Results/brando01/sequence\(index)/brando01\(i).png", bbox_inches: "tight")
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
      fig.savefig("Results/brando01/brando01_subsequence\(index).png", bbox_inches: "tight")
      print("Accuracy for sequence is \(value.sequenceMetrics.accuracy) with Robustness of \(value.sequenceMetrics.robustness)")
    }

    print("Accuracy for all sequences is \(results.trackerMetrics.accuracy) with Robustness of \(results.trackerMetrics.robustness)")
    


  }

}