import ArgumentParser
import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation
import PenguinStructures

/// Brando12: OPTIMIZATION CONVERGENCE VISUALIZATION - RAE + MVG
struct Brando13: ParsableCommand {
  @Option(help: "Run for number of frames")
  var trackLength: Int = 80

  func run() {
    let np = Python.import("numpy")
    let plt = Python.import("matplotlib.pyplot")
    let trainingDatasetSize = 100

    // LOAD THE IMAGE AND THE GROUND TRUTH ORIENTED BOUNDING BOX
    let dataDir = URL(fileURLWithPath: "./OIST_Data")
    let data = OISTBeeVideo(directory: dataDir, length: trainingDatasetSize)!
    let testData = OISTBeeVideo(directory: dataDir, afterIndex: trainingDatasetSize, length: trackLength)!
    let frames = testData.frames
    let firstTrack = testData.tracks[0]
    // let firstTrack = testData.tracks[5]
    let firstFrame = frames[0]
    let firstObb = firstTrack.boxes[0]
    // let firstObb = firstTrack.boxes[5]


    // CREATE A PLACEHOLDER FOR POSE
    var v = VariableAssignments()


    // LOAD THE CLASSIFIER
    let (imageHeight, imageWidth, imageChannels) =
      (40, 70, 1)
    let featureSize = 256
    let kHiddenDimension = 512
    var rae = DenseRAE(
      imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels,
      hiddenDimension: kHiddenDimension, latentDimension: featureSize
    )
    rae.load(weights: np.load("./oist_rae_weight_\(featureSize).npy", allow_pickle: true))
    print("hello2")
    let (fg, bg, _) = getTrainingBatches(
        dataset: data, boundingBoxSize: (40, 70),
        fgBatchSize: 3000,
        bgBatchSize: 3000,
        fgRandomFrameCount: 10,
        bgRandomFrameCount: 10,
        useCache: true
    )
    let batchPositive = rae.encode(fg)
    // let foregroundModel = GaussianNB(from:batchPositive, regularizer: 1e-3)
    let foregroundModel = MultivariateGaussian(from:batchPositive, regularizer: 1e-3)


    let batchNegative = rae.encode(bg)
    // let backgroundModel = GaussianNB(from: batchNegative, regularizer: 1e-3)
    let backgroundModel = MultivariateGaussian(from: batchNegative, regularizer: 1e-3)
        
    

    

    //OPTIMIZER GRADIENT DESCENT
    let lr = 1e-5
    var optimizer = GradientDescent(learningRate: lr)

    //CREATE A FOLDER TO CONTAIN THE END-RESULT IMAGES OF THE OPTIMIZATION
    let folderName = "Results/GD_optimization_RAE_\(lr)_final_images"
      if !FileManager.default.fileExists(atPath: folderName) {
      do {
          try FileManager.default.createDirectory(atPath: folderName, withIntermediateDirectories: true, attributes: nil)
      } catch {
          print(error.localizedDescription)
      }
      }

    //CREATE A FIG
    print("hello1")
    let (fig, axs) = plt.subplots(2,2).tuple2
    let fr = np.squeeze(firstFrame.makeNumpyArray())
    for i in 0...1 {
      for j in 0...1 {
        axs[i,j].imshow(fr / 255.0, cmap: "gray")
        let firstGroundTruth = firstObb.center
        // axs[i,j].plot(firstObb.corners.map{$0.x} + [firstObb.corners.first!.x], firstObb.corners.map{$0.y} + [firstObb.corners.first!.y], "b-")
        axs[i,j].set_xlim(firstGroundTruth.t.x - 50, firstGroundTruth.t.x + 50)
        axs[i,j].set_ylim(firstGroundTruth.t.y - 50, firstGroundTruth.t.y + 50)
        axs[i,j].get_xaxis().set_visible(false)
        axs[i,j].get_yaxis().set_visible(false)
      }
    }
    axs[0,0].set_title("fabs(theta) < 0.1", fontsize:8)
    axs[0,1].set_title("fabs(theta) < 0.2", fontsize:8)
    axs[1,0].set_title("fabs(theta) < 0.3", fontsize:8)
    axs[1,1].set_title("fabs(theta) >= 0.3", fontsize:8)

    print("hello")
    let xy_thresh = 20.0 //pixels
    let theta_thresh = 0.5 //radians // consider doing overlap.

    //PERFORM THIS OPTIMIZATION J TIMES
    for j in 0..<200 {

      // RANDOMLY PERTURB THE GROUND TRUTH POSE AND CALCULATE THE PERTURBATION
      let poseId = v.store(firstObb.center)
      v[poseId].perturbWith(stddev: Vector3(0.3, 8, 4.6))
      let dx = v[poseId].t.x - firstObb.center.t.x
      let dy = v[poseId].t.y - firstObb.center.t.y
      let dtheta = v[poseId].rot.theta - firstObb.center.rot.theta
      let startpose = v[poseId]

      // CREATE THE FACTOR AND FACTOR GRAPH
      var fg = FactorGraph()
      let factor = ProbablisticTrackingFactor(poseId,
        measurement: firstFrame,
        encoder: rae,
        patchSize: (40, 70),
        appearanceModelSize: (40, 70),
        foregroundModel: foregroundModel,
        backgroundModel: backgroundModel,
        maxPossibleNegativity: 1e10
      )
      fg.store(factor)

      let it_limit = 1000


      

      // PERFORM GRADIENT DESCENT
      var conv = true
      var errors = [Double]()

      for i in 0..<it_limit {
        errors.append(factor.errorVector(v[poseId]).x)
        print("iteration \(i) error:", factor.errorVector(v[poseId]).x, "x:", v[poseId].t.x, "y:", v[poseId].t.y, "theta:", v[poseId].rot.theta)
        let oldpose = v[poseId]
        optimizer.update(&v, objective: fg)
        // WHEN DIFF IS SO SMALL, THE OPTIMIZATION HAS CONVERGED
        if abs(v[poseId].t.x - oldpose.t.x) < 0.000001 && abs(v[poseId].t.y - oldpose.t.y) < 0.000001 && abs(v[poseId].rot.theta - oldpose.rot.theta) < 0.000001{
          print("converged on iteration number \(i). Final Error:", factor.errorVector(v[poseId]), "Initial error:", factor.errorVector(startpose))
          break
        }
        if i == it_limit-1 {
          conv = false
          print("no convergence :( Final Error:", factor.errorVector(v[poseId]), "Initial error:", factor.errorVector(startpose))
        }
      }
      // PLOT THE FINAL OPTIMIZATION RESULT
      let x_out_of_bounds = (v[poseId].t.x > firstObb.center.t.x + xy_thresh) || (v[poseId].t.x < firstObb.center.t.x - xy_thresh)
      let y_out_of_bounds =  (v[poseId].t.y > firstObb.center.t.y + xy_thresh) || (v[poseId].t.y < firstObb.center.t.y - xy_thresh)
      let theta_out_of_bounds = (v[poseId].rot.theta > firstObb.center.rot.theta + theta_thresh) || (v[poseId].rot.theta < firstObb.center.rot.theta - theta_thresh)
        if !x_out_of_bounds && !theta_out_of_bounds && !y_out_of_bounds {
            // plot a green dot
            // ax.scatter(startpose.t.x-Double(xbegin),startpose.t.y-Double(ybegin),c:"r", marker: ",")
            // ax.scatter(startpose.t.x,startpose.t.y,c:"r", marker: ",")
            if fabs(startpose.rot.theta - firstObb.center.rot.theta) < 0.1 {
                axs[0,0].plot(startpose.t.x,startpose.t.y,"g,", ms: 1)
            } else if fabs(startpose.rot.theta - firstObb.center.rot.theta) < 0.2 {
                axs[0,1].plot(startpose.t.x,startpose.t.y,"g,", ms: 1)
            } else if fabs(startpose.rot.theta - firstObb.center.rot.theta) < 0.3 {
                axs[1,0].plot(startpose.t.x,startpose.t.y,"g,", ms: 1)
            } else {
                axs[1,1].plot(startpose.t.x,startpose.t.y,"g,", ms: 1)
            }
            
        } else {
            // ax.scatter(startpose.t.x-Double(xbegin),startpose.t.y-Double(ybegin),c:"g", marker: ",")
            // ax.scatter(startpose.t.x,startpose.t.y,c:"g", marker: ",")
            if fabs(startpose.rot.theta - firstObb.center.rot.theta) < 0.1 {
                axs[0,0].plot(startpose.t.x,startpose.t.y,"r,", ms: 1)
            } else if fabs(startpose.rot.theta - firstObb.center.rot.theta) < 0.2 {
                axs[0,1].plot(startpose.t.x,startpose.t.y,"r,", ms: 1)
            } else if fabs(startpose.rot.theta - firstObb.center.rot.theta) < 0.3 {
                axs[1,0].plot(startpose.t.x,startpose.t.y,"r,", ms: 1)
            } else {
                axs[1,1].plot(startpose.t.x,startpose.t.y,"r,", ms: 1)
            }
        }
        let (figs, axes) = plotFrameWithPatches3(frame: firstFrame, start: startpose, end: v[poseId], expected: firstObb.center, firstGroundTruth: firstObb.center, errors: errors)
        axes.set_title(String(axes.get_title())! + "\n final err = \(factor.errorVector(v[poseId]).x)" 
        + "\n label err = \(factor.errorVector(firstObb.center).x)" 
        + "\n start err = \(factor.errorVector(startpose).x)"
        + "\n learning rate = \(lr)"
        + "\n converged = \(conv)")
        figs.savefig(folderName + "/optimization_final_\(j).png", bbox_inches: "tight")
        plt.close("all")
        fig.savefig(folderName + "/optimization_covergence_red_n_green_dots.png", bbox_inches: "tight")

    }
    print("done")
  }
}