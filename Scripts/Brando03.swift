import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation

import PenguinStructures

/// Brando01 SiamMask
struct Brando03: ParsableCommand {
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
    // let data = OISTBeeVideo(directory: dataDir, length: 100)!
    let testData = OISTBeeVideo(directory: dataDir, afterIndex: 100, length: 80)!
    // print("number of frames in training data:", data.labels.count)
    print("number of frames in testing data", testData.labels.count, "\n\n")


    let trackerEvaluation = TrackerEvaluationDataset(testData)
    // let shpl = Python.import("shapely")
    let os = Python.import("os")

    // print(os.environ)
    // let plt = Python.import("matplotlib")
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
    // parser.add_argument("--cpu")
    // let args = parser.parse_args(["--resume", "../SiamMask/experiments/siammask_sharp/SiamMask_VOT.pth", "--config", "../SiamMask/experiments/siammask_sharp/config_vot.json", "--base_path", "./OIST_Data/downsampled"])
    // let args = parser.parse_args(["--resume", "../SiamMask/checkpoint_e20.pth", "--config", "../SiamMask/experiments/siammask_sharp/config_vot.json", "--base_path", "./OIST_Data/downsampled"])
    let args = parser.parse_args(["--resume", "../SiamMask/model_sharp/checkpoint_e20.pth", "--config", "../SiamMask/experiments/siammask_sharp/config_vot.json", "--base_path", "./OIST_Data/downsampled"])

    print("ARGUMENTS", args)


    // let imutils = Python.import("utils")
    print(Python.version)
    print("hello")
    let evalTracker: Tracker = { frames, start in

        //SIAM MASK TRACKER IS HERE
        let device = torch.device("cpu")
        torch.backends.cudnn.benchmark = true

        // # Setup Model
        let cfg = cfhelper.load_config(args)
        let custom = Python.import("SiamMask.experiments.siammask_sharp.custom")
        // // from custom import Custom
        var siammask = custom.Custom(anchors: cfg["anchors"])
        // if args.resume:
        // assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = ldhelper.load_pretrain(siammask, args.resume)

        siammask.eval().to(device)

        // # Parse Image file
        // img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
        // ims = [cv2.imread(imf) for imf in img_files]

        // # Select ROI
        // cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
        // # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        // try:
        //     init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
        let init_rect = Python.tuple([Int(start.center.t.x)-20, Int(start.center.t.y)-20, 40, 70])
        let tup = init_rect.tuple4
        let x = tup.0
        let y = tup.1
        let w = tup.2
        let h = tup.3
        //     x, y, w, h = init_rect
        // // except:
        // //     exit()

        // var toc = 0
        var state: PythonObject = 0
        var results = [PythonObject]()

        for (f, im) in frames.enumerated() {
        // for f, im in enumerate(ims):
            // let tic = cv2.getTickCount()
            let im_np = im.makeNumpyArray()
            let im_3d = np.squeeze(np.stack(Python.tuple([im_np, im_np, im_np]), axis: 2))
            // print("image shape", im_3d.shape)
            // cv2.imshow("SiamMask", im_3d)
            if f == 0 { // init
                let target_pos = np.array([x + w / 2, y + h / 2])
                let target_sz = np.array([w, h])
                state = smtest.siamese_init(im_3d, target_pos, target_sz, siammask, cfg["hp"], device: device)  //# init tracker
                results.append(Python.tuple([Int(x + w / 2)!, Int(y + h / 2)!]))
            } else if f > 0 {  //# tracking
                state = smtest.siamese_track(state, im_3d, mask_enable: true, refine_enable: true, device: device)  //# track
                let location = state["ploygon"].flatten()
                
                // cv2.polylines(im_3d, [np.int0(location).reshape(Python.tuple([-1, 1, 2]))], true, Python.tuple([0,255,0]), 3)
                // cv2.circle(im_3d, Python.tuple([centx, centy]), 10, Python.tuple([0,255,255]), 5)
                // cv2.imwrite("SiamMask"+String(f)+".png", im_3d)
                // let mask = state["mask"] > state["p"].seg_thr
                results.append(location)
                // im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]

                
            }
            
        }
        // results.map{print($0)}
        // print("hello")
        // print(type(of: results))
        // print(results)
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
            // if locx >= centx && locy >= centy{}
            
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
            // let vect = Vector2(Double(pythonBB.0)! + Double(rows)/2, Double(pythonBB.1)! + Double(cols)/2)
            print("rotation", rot, "\n\n")
            let center = Pose2(rot, vect)
            let swiftBB = OrientedBoundingBox(center: center, rows: rows, cols: cols)
            track.append(swiftBB)
          } else {
            let swiftBB = start
            track.append(swiftBB)
          }
        }
        // print(track)
        return track
    }

    let plt = Python.import("matplotlib.pyplot")
    let sequenceCount = 20
    var eval_results = trackerEvaluation.evaluate(evalTracker, sequenceCount: sequenceCount, deltaAnchor: 175, outputFile: "brando03")
    // print(results)
    print("done evaluating")
    var total_overlap = eval_results.sequences.prefix(sequenceCount)[0].subsequences.first!.metrics.overlap
    // total_overlap += eval_results.sequences.prefix(sequenceCount)[1].subsequences.first!.metrics.overlap

    for (index, value) in eval_results.sequences.prefix(sequenceCount).enumerated() {
      // var i: Int = 0
      // zip(value.subsequences.first!.frames, zip(value.subsequences.first!.prediction, value.subsequences.first!.groundTruth)).map {
      //   let (fig, axes) = plotFrameWithPatches(frame: $0.0, actual: $0.1.0.center, expected: $0.1.1.center, firstGroundTruth: value.subsequences.first!.groundTruth.first!.center)
      //   fig.savefig("Results/brando03/sequence\(index)/brando03\(i).png", bbox_inches: "tight")
      //   plt.close("all")
      //   i = i + 1
      // }
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