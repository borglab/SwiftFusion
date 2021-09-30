import ArgumentParser
import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation
import PenguinStructures

/// Brando12: OPTIMIZATION CONVERGENCE VISUALIZATION
struct Brando12: ParsableCommand {
  @Option(help: "Run for number of frames")
  var trackLength: Int = 80

  @Option(help: "Classifier or rae")
  var useClassifier: Bool = true

  func initialize_and_perturb(p: Pose2) -> (Double, Double, Double, Pose2, VariableAssignments, TypedID<Pose2>, FactorGraph) {
    // CREATE A PLACEHOLDER FOR POSE
    var v = VariableAssignments()
    let poseId = v.store(p)
    v[poseId].perturbWith(stddev: Vector3(0.3, 8, 4.6))
    let dx = v[poseId].t.x - p.t.x
    let dy = v[poseId].t.y - p.t.y
    let dtheta = v[poseId].rot.theta - p.rot.theta
    let startpose = v[poseId]
    let fg = FactorGraph()

    return (dx, dy, dtheta, startpose, v, poseId, fg)
  }

  func initialize_empty_arrays() -> (Bool, [Double], [Double], [Double], [Double]) {
    var conv = true
    var errors = [Double]()
    var xs = [Double]()
    var ys = [Double]()
    var thetas = [Double]()
    return (conv, errors, xs, ys, thetas)
  }



  func run() {
    let np = Python.import("numpy")
    let plt = Python.import("matplotlib.pyplot")
    let trainingDatasetSize = 100

    // LOAD THE IMAGE AND THE GROUND TRUTH ORIENTED BOUNDING BOX
    let dataDir = URL(fileURLWithPath: "./OIST_Data")
    let testData = OISTBeeVideo(directory: dataDir, afterIndex: trainingDatasetSize, length: trackLength)!
    let data = OISTBeeVideo(directory: dataDir, length: trainingDatasetSize)!
    let frames = testData.frames
    let firstTrack = testData.tracks[0]
    // let firstTrack = testData.tracks[5]
    let firstFrame = frames[0]
    let firstObb = firstTrack.boxes[0]
    // let firstObb = firstTrack.boxes[5]
      

    //OPTIMIZER GRADIENT DESCENT
    let lr = 1e-7
    var optimizer = GradientDescent(learningRate: lr)
    let it_limit = 200


    //CREATE A FOLDER TO CONTAIN THE END-RESULT IMAGES OF THE OPTIMIZATION
    let str: String
    if useClassifier{
      str = "NNC"
    } else {
      str = "RAE"
    }
    let folderName = "Results/GD_optimization_\(str)_lr_\(lr)__3_09_2021_final_images_4subplots"
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

    
    // NN Params
    let (imageHeight, imageWidth, imageChannels) = (40, 70, 1)
    let featureSize = 256
    let kHiddenDimension = 512


    if useClassifier {
      var classifier = NNClassifier(
        imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels, hiddenDimension: kHiddenDimension, latentDimension: featureSize
      )
      classifier.load(weights: np.load("./classifiers/classifiers_today/classifier_weight_\(kHiddenDimension)_\(featureSize)_1_doubletraining.npy", allow_pickle: true))
      
      for j in 0...200 {
        // RANDOMLY PERTURB THE GROUND TRUTH POSE AND CALCULATE THE PERTURBATION
        var (dx, dy, dtheta, startpose, v, poseId, fg) = initialize_and_perturb(p: firstObb.center)
        // CREATE THE FACTOR AND FACTOR GRAPH
        let factorNNC = ProbablisticTrackingFactor2(poseId,
          measurement: firstFrame,
          classifier: classifier,
          patchSize: (40, 70),
          appearanceModelSize: (40, 70)
        )
        fg.store(factorNNC)


        // PERFORM GRADIENT DESCENT
      var (conv, errors, xs, ys, thetas) = initialize_empty_arrays()
      print("starting optimization")
      for i in 0..<it_limit {
        errors.append(factorNNC.errorVector(v[poseId]).x)
        xs.append(v[poseId].t.x)
        ys.append(v[poseId].t.y)
        thetas.append(v[poseId].rot.theta)
        // print("iteration \(i) error:", factor.errorVector(v[poseId]).x, "x:", v[poseId].t.x, "y:", v[poseId].t.y, "theta:", v[poseId].rot.theta)
        let oldpose = v[poseId]
        optimizer.update(&v, objective: fg)
        // WHEN DIFF IS SO SMALL, THE OPTIMIZATION HAS CONVERGED
        if abs(v[poseId].t.x - oldpose.t.x) < 0.000001 && abs(v[poseId].t.y - oldpose.t.y) < 0.000001 && abs(v[poseId].rot.theta - oldpose.rot.theta) < 0.000001{
          // print("converged on iteration number \(i). Final Error:", factor.errorVector(v[poseId]), "Initial error:", factor.errorVector(startpose))
          break
        }
        if i == it_limit-1 {
          conv = false
          // print("no convergence :( Final Error:", factor.errorVector(v[poseId]), "Initial error:", factor.errorVector(startpose))
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
      let (figs, axes) = plotFrameWithPatches3(frame: firstFrame, start: startpose, end: v[poseId], expected: firstObb.center, firstGroundTruth: firstObb.center, errors: errors, xs: xs, ys: ys, thetas: thetas)
      var final_err: Double
      var label_err: Double
      var start_err: Double

      
      final_err = factorNNC.errorVector(v[poseId]).x
      label_err = factorNNC.errorVector(firstObb.center).x
      start_err = factorNNC.errorVector(startpose).x

      axes.set_title(String(axes.get_title())! + "\n final err = \(final_err)" 
      + "\n label err = \(label_err).x)" 
      + "\n start err = \(start_err)"
      + "\n learning rate = \(lr)"
      + "\n converged = \(conv)")
      figs.savefig(folderName + "/optimization_final_\(j).png", bbox_inches: "tight")
      // let (figs2, axes2) = plotXYandTheta(xs: xs, ys: ys, thetas: thetas)
      // figs2.savefig(folderName + "/optimization_final_\(j)_XYtheta.png", bbox_inches: "tight")
      plt.close("all")
      fig.savefig(folderName + "/optimization_covergence_red_n_green_dots.png", bbox_inches: "tight")

      

      
      }
        

      



    } else {
      // LOAD RAE AND TRAIN BG AND FG MODELS
      var rae = DenseRAE(
      imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels,
      hiddenDimension: kHiddenDimension, latentDimension: featureSize
      )
      rae.load(weights: np.load("./oist_rae_weight_\(featureSize).npy", allow_pickle: true))
      let (fg, bg, _) = getTrainingBatches(
          dataset: data, boundingBoxSize: (40, 70), fgBatchSize: 3000, bgBatchSize: 3000,
          fgRandomFrameCount: 10, bgRandomFrameCount: 10, useCache: true
      )
      let batchPositive = rae.encode(fg)
      let foregroundModel = MultivariateGaussian(from:batchPositive, regularizer: 1e-3)
      let batchNegative = rae.encode(bg)
      let backgroundModel = MultivariateGaussian(from: batchNegative, regularizer: 1e-3)
      
      for j in 0...200 {
        
        // RANDOMLY PERTURB THE GROUND TRUTH POSE AND CALCULATE THE PERTURBATION
        var (dx, dy, dtheta, startpose, v, poseId, fg) = initialize_and_perturb(p: firstObb.center)
        // CREATE THE FACTOR AND FACTOR GRAPH
        let factorRAE = ProbablisticTrackingFactor(poseId,
            measurement: firstFrame,
            encoder: rae,
            patchSize: (40, 70),
            appearanceModelSize: (40, 70),
            foregroundModel: foregroundModel,
            backgroundModel: backgroundModel,
            maxPossibleNegativity: 1e7
        )
        fg.store(factorRAE)
        // PERFORM GRADIENT DESCENT
        var (conv, errors, xs, ys, thetas) = initialize_empty_arrays()
        print("starting optimization")
        for i in 0..<it_limit {
          errors.append(factorRAE.errorVector(v[poseId]).x)
          xs.append(v[poseId].t.x)
          ys.append(v[poseId].t.y)
          thetas.append(v[poseId].rot.theta)
          // print("iteration \(i) error:", factor.errorVector(v[poseId]).x, "x:", v[poseId].t.x, "y:", v[poseId].t.y, "theta:", v[poseId].rot.theta)
          let oldpose = v[poseId]
          optimizer.update(&v, objective: fg)
          // WHEN DIFF IS SO SMALL, THE OPTIMIZATION HAS CONVERGED
          if abs(v[poseId].t.x - oldpose.t.x) < 0.000001 && abs(v[poseId].t.y - oldpose.t.y) < 0.000001 && abs(v[poseId].rot.theta - oldpose.rot.theta) < 0.000001{
            // print("converged on iteration number \(i). Final Error:", factor.errorVector(v[poseId]), "Initial error:", factor.errorVector(startpose))
            break
          }
          if i == it_limit-1 {
            conv = false
            // print("no convergence :( Final Error:", factor.errorVector(v[poseId]), "Initial error:", factor.errorVector(startpose))
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
          let (figs, axes) = plotFrameWithPatches3(frame: firstFrame, start: startpose, end: v[poseId], expected: firstObb.center, firstGroundTruth: firstObb.center, errors: errors, xs: xs, ys: ys, thetas: thetas)
          var final_err: Double
          var label_err: Double
          var start_err: Double

          final_err = factorRAE.errorVector(v[poseId]).x
          label_err = factorRAE.errorVector(firstObb.center).x
          start_err = factorRAE.errorVector(startpose).x
        
          axes.set_title(String(axes.get_title())! + "\n final err = \(final_err)" 
          + "\n label err = \(label_err).x)" 
          + "\n start err = \(start_err)"
          + "\n learning rate = \(lr)"
          + "\n converged = \(conv)")
          figs.savefig(folderName + "/optimization_final_\(j).png", bbox_inches: "tight")
          // let (figs2, axes2) = plotXYandTheta(xs: xs, ys: ys, thetas: thetas)
          // figs2.savefig(folderName + "/optimization_final_\(j)_XYtheta.png", bbox_inches: "tight")
          plt.close("all")
          fig.savefig(folderName + "/optimization_covergence_red_n_green_dots.png", bbox_inches: "tight")

      }
    }
  }
}