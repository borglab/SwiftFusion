import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation

import PenguinStructures

/// Brando01 SiamMask Tracker
struct Brando03: ParsableCommand {

  func run() {

    let dataDir = URL(fileURLWithPath: "./OIST_Data")
    let testData = OISTBeeVideo(directory: dataDir, afterIndex: 100, length: 80)!
    print("number of frames in testing data", testData.labels.count, "\n\n")


    let trackerEvaluation = TrackerEvaluationDataset(testData)
    let os = Python.import("os")
    let torch = Python.import("torch")

    let np = Python.import("numpy")
    let smtools = Python.import("SiamMask.tools")
    let smutils = Python.import("SiamMask.utils")
    let cfhelper = Python.import("SiamMask.utils.config_helper")
    let ldhelper = Python.import("SiamMask.utils.load_helper")
    let smtest = Python.import("SiamMask.tools.test")
    

    let cv2 = Python.import("cv2")
  
    let argparse = Python.import("argparse")
    let parser = argparse.ArgumentParser()
    
    parser.add_argument("--resume")
    parser.add_argument("--config")
    parser.add_argument("--base_path")
    let args = parser.parse_args(["--resume", "../SiamMask/model_sharp/checkpoint_e20.pth", "--config", "../SiamMask/experiments/siammask_sharp/config_vot.json", "--base_path", "./OIST_Data/downsampled"])

    print("ARGUMENTS", args)


    print(Python.version)
    print("hello")
    let evalTracker: Tracker = { frames, start in

        //SIAM MASK TRACKER IS HERE
        let device = torch.device("cpu")
        torch.backends.cudnn.benchmark = true

        // # Setup Model
        let cfg = cfhelper.load_config(args)
        let custom = Python.import("SiamMask.experiments.siammask_sharp.custom")
        var siammask = custom.Custom(anchors: cfg["anchors"])
        siammask = ldhelper.load_pretrain(siammask, args.resume)

        siammask.eval().to(device)
        let init_rect = Python.tuple([Int(start.center.t.x)-20, Int(start.center.t.y)-20, 40, 70])
        let tup = init_rect.tuple4
        let x = tup.0
        let y = tup.1
        let w = tup.2
        let h = tup.3

        var state: PythonObject = 0
        var results = [PythonObject]()

        for (f, im) in frames.enumerated() {

            let im_np = im.makeNumpyArray()
            let im_3d = np.squeeze(np.stack(Python.tuple([im_np, im_np, im_np]), axis: 2))

            if f == 0 { // init
                let target_pos = np.array([x + w / 2, y + h / 2])
                let target_sz = np.array([w, h])
                state = smtest.siamese_init(im_3d, target_pos, target_sz, siammask, cfg["hp"], device: device)  //# init tracker
                results.append(Python.tuple([Int(x + w / 2)!, Int(y + h / 2)!]))
            } else if f > 0 {  //# tracking
                state = smtest.siamese_track(state, im_3d, mask_enable: true, refine_enable: true, device: device)  //# track
                let location = state["ploygon"].flatten()
                

                results.append(location)

                
            }
            
        }

        var track = [OrientedBoundingBox]()
        for (i, result) in results.enumerated() {
          if i > 0 {
            let location = result
            let centx = Int((location[0]+location[2]+location[4]+location[6])/4)!
            let centy = Int((location[1]+location[3]+location[5]+location[7])/4)!
            let dx1 = location[0]-location[2]
            let dy1 = location[1]-location[3]
            let dx2 = location[0]-location[6]
            let dy2 = location[1]-location[7]
            let dist1 = sqrt(pow(Double(dx1)!, 2) + pow(Double(dy1)!, 2))
            let dist2 = (pow(Double(dx2)!, 2) + pow(Double(dy2)!, 2)).squareRoot()
            let locx: Int
            let locy: Int
            let rows: Int
            let cols: Int
            if dist1 < dist2 {
                locx = Int((location[0]+location[2])/2)!
                locy = Int((location[1]+location[3])/2)!
                rows = Int(dist1)
                cols = Int(dist2)
            } else {
                locx = Int((location[0]+location[6])/2)!
                locy = Int((location[1]+location[7])/2)! 
                rows = Int(dist2)
                cols = Int(dist1)
            }
            let dx = Double(abs(locx - centx))
            let dy = Double(abs(locy - centy))
            var theta = Double.pi/2
            print("polygon", result)
            print("center", centx, centy)
            print("dx and dy",  dx, dy)
            print("theta initial", theta)
            if dx != 0 {
                theta = atan(dy/dx)
            }
            
            if locx >= centx && locy < centy{
                theta = -theta
            } else if locx < centx && locy >= centy{
                theta = .pi - theta
            } else if locx < centx && locy < centy{
                theta = .pi + theta
            }
            print("theta final", theta)

            let rot = Rot2(theta)
            let vect = Vector2(Double(centx), Double(centy))
            print("rotation", rot, "\n\n")
            let center = Pose2(rot, vect)
            let swiftBB = OrientedBoundingBox(center: center, rows: rows, cols: cols)
            track.append(swiftBB)
          } else {
            let swiftBB = start
            track.append(swiftBB)
          }
        }
        return track
    }

    let plt = Python.import("matplotlib.pyplot")
    let sequenceCount = 20
    var eval_results = trackerEvaluation.evaluate(evalTracker, sequenceCount: sequenceCount, deltaAnchor: 175, outputFile: "brando03")
    print("done evaluating")
    var total_overlap = eval_results.sequences.prefix(sequenceCount)[0].subsequences.first!.metrics.overlap

    for (index, value) in eval_results.sequences.prefix(sequenceCount).enumerated() {

      print("done,", index)      
      let (fig, axes) = plt.subplots(1, 2, figsize: Python.tuple([20, 20])).tuple2
      fig.suptitle("Tracking positions and Subsequence Average Overlap with Accuracy \(String(format: "%.2f", value.subsequences.first!.metrics.accuracy)) and Robustness \(value.subsequences.first!.metrics.robustness).")
      value.subsequences.map {
        //zip($0.prediction, $0.groundTruth).enumerated().map{($0.0, $0.1.0.center, $0.1.1.center)})
        let encoder = JSONEncoder()
        let data = try! encoder.encode($0.prediction)
        FileManager.default.createFile(atPath: "Results/brando03/prediction_siammask_sequence_\(index).json", contents: data, attributes: nil)
      }
      value.subsequences.map {
        plotPoseDifference(
          track: $0.prediction.map{$0.center}, withGroundTruth: $0.groundTruth.map{$0.center}, on: axes[0]
        )
      }
      plotOverlap(
          metrics: value.subsequences.first!.metrics, on: axes[1]
      )

      fig.savefig("Results/brando03/brando03_subsequence\(index).png", bbox_inches: "tight")
      print("Accuracy for sequence is \(value.sequenceMetrics.accuracy) with Robustness of \(value.sequenceMetrics.robustness)")
    }
    print("Accuracy for all sequences is \(eval_results.trackerMetrics.accuracy) with Robustness of \(eval_results.trackerMetrics.robustness)")
    
    let pickle = Python.import("pickle");
    let f = Python.open("Results/EAO/siammask.data", "wb")
    pickle.dump(eval_results.expectedAverageOverlap.curve, f)

    
    // var average_overlap = [Double]()
    // for (i, val) in total_overlap.enumerated() {
    //   average_overlap.append(val/Double(sequenceCount))
    // }
    // let (fig, ax) = plt.subplots().tuple2
    // ax.plot(average_overlap)
    // ax.set_title("Overlap")
    // fig.savefig("average_overlap.png")



    

  }

}