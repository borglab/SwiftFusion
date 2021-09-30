import ArgumentParser
import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation
import PenguinStructures

/// Brando09: OPTIMIZATION VISUALIZATION
struct Brando09: ParsableCommand {
  @Option(help: "Run for number of frames")
  var trackLength: Int = 80

  func run() {
//     let np = Python.import("numpy")
//     let plt = Python.import("matplotlib.pyplot")
//     let trainingDatasetSize = 100

//     // LOAD THE IMAGE AND THE GROUND TRUTH ORIENTED BOUNDING BOX
//     let dataDir = URL(fileURLWithPath: "./OIST_Data")
//     let testData = OISTBeeVideo(directory: dataDir, afterIndex: trainingDatasetSize, length: trackLength)!
//     let frames = testData.frames
//     let firstTrack = testData.tracks[0]
//     // let firstTrack = testData.tracks[5]
//     let firstFrame = frames[0]
//     let firstObb = firstTrack.boxes[0]
//     // let firstObb = firstTrack.boxes[5]


//     // CREATE A PLACEHOLDER FOR POSE
//     var v = VariableAssignments()


//     // LOAD THE CLASSIFIER
//     let (imageHeight, imageWidth, imageChannels) =
//       (40, 70, 1)
//     let featureSize = 512
//     let kHiddenDimension = 512
//     // var classifier = SmallerNNClassifier(
//     //   imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels, latentDimension: featureSize
//     // )
//     var classifier = NNClassifier(
//       imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels, hiddenDimension: kHiddenDimension, latentDimension: featureSize
//     )
//     // classifier.load(weights: np.load("./classifiers/classifiers_today/small_classifier_weight_\(featureSize)_2.npy", allow_pickle: true))
//     classifier.load(weights: np.load("./classifiers/classifiers_today/classifier_weight_\(kHiddenDimension)_\(featureSize)_1_doubletraining.npy", allow_pickle: true))
    
    

//     //OPTIMIZER GRADIENT DESCENT
//     let lr = 1e-4
//     var optimizer = GradientDescent(learningRate: lr)

//     //CREATE A FOLDER TO CONTAIN THE END-RESULT IMAGES OF THE OPTIMIZATION
//     let folderName = "Results/GD_optimization_lr_\(lr)_final_images"
//       if !FileManager.default.fileExists(atPath: folderName) {
//       do {
//           try FileManager.default.createDirectory(atPath: folderName, withIntermediateDirectories: true, attributes: nil)
//       } catch {
//           print(error.localizedDescription)
//       }
//       }

//     //PERFORM THIS OPTIMIZATION J TIMES
//     for j in 0..<20 {
    
//       // RANDOMLY PERTURB THE GROUND TRUTH POSE AND CALCULATE THE PERTURBATION
//       let poseId = v.store(firstObb.center)
//       v[poseId].perturbWith(stddev: Vector3(0.3, 8, 4.6))
//       let dx = v[poseId].t.x - firstObb.center.t.x
//       let dy = v[poseId].t.y - firstObb.center.t.y
//       let dtheta = v[poseId].rot.theta - firstObb.center.rot.theta
//       let startpose = v[poseId]

//       // CREATE THE FACTOR AND FACTOR GRAPH
//       var fg = FactorGraph()
//       let factor = ProbablisticTrackingFactor2(poseId,
//         measurement: firstFrame,
//         classifier: classifier,
//         patchSize: (40, 70),
//         appearanceModelSize: (40, 70)
//       )
//       fg.store(factor)


//       // CREATE A FOLDER FOR EACH OPTIMIZATION ROUND. 
//       // let folderName = "Results/GD_optimization_lr_\(lr)_\(j)"
//       //   if !FileManager.default.fileExists(atPath: folderName) {
//       //   do {
//       //       try FileManager.default.createDirectory(atPath: folderName, withIntermediateDirectories: true, attributes: nil)
//       //   } catch {
//       //       print(error.localizedDescription)
//       //   }
//       //   }

//       // MAX ITERATIONS FOR OPTIMIZATION
//       let it_limit = 1000
//       print("\(j)) Starting Optimization from: \(dx), \(dy), \(dtheta)")
      

//       // PERFORM GRADIENT DESCENT
//       for i in 0..<it_limit {
//             print("iteration \(i) error:", factor.errorVector(v[poseId]).norm)
//             let oldpose = v[poseId]

//             // PLOT THE OPTIMIZATION BOUNDING BOX ON EACH ITERATION OF THE OPTIMIZATION
//             // let (fig, axes) = plotFrameWithPatches3(frame: firstFrame, start: startpose, end: v[poseId], expected: firstObb.center, firstGroundTruth: firstObb.center)
//             // axes.set_title(String(axes.get_title())! + " index_\(i)")
//             // fig.savefig(folderName + "/optimization_index_\(i).png", bbox_inches: "tight")
//             // plt.close("all")

//             optimizer.update(&v, objective: fg)
            

//             // WHEN DIFF IS SO SMALL, THE OPTIMIZATION HAS CONVERGED
//             if abs(v[poseId].t.x - oldpose.t.x) < 0.0001 && abs(v[poseId].t.y - oldpose.t.y) < 0.0001 && abs(v[poseId].rot.theta - oldpose.rot.theta) < 0.0001{
//               print("converged on iteration number \(i). Final Error:", factor.errorVector(v[poseId]))
//               break
//             }
//             if i == it_limit-1 {
//               print("no convergence :( Final Error:", factor.errorVector(v[poseId]))
//             }
            
          
//       }
      
//       // PLOT THE FINAL OPTIMIZATION RESULT
//       let (fig, axes) = plotFrameWithPatches3(frame: firstFrame, start: startpose, end: v[poseId], expected: firstObb.center, firstGroundTruth: firstObb.center)
//       axes.set_title(String(axes.get_title())! + "\n err = \(factor.errorVector(v[poseId]).norm)")
//       fig.savefig(folderName + "/optimization_final_\(j).png", bbox_inches: "tight")
//       plt.close("all")
//     }
    
//     print("done")

  }
}
