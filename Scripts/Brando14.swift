// NN Classifier
// Load 1st image
// Load Classifier
// take the error value at each pixel in cropped image
// plot the error value on the image from white to red

import ArgumentParser
import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation
import PenguinStructures

/// Brando12: OPTIMIZATION CONVERGENCE VISUALIZATION
struct Brando14: ParsableCommand {
  @Option(help: "Run for number of frames")
  var trackLength: Int = 80

  @Option(help: "Classifier or rae")
  var useClassifier: Bool = true


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

    let range = 100.0
      
    // NN Params
    let (imageHeight, imageWidth, imageChannels) = (40, 70, 1)
    let featureSize = 256
    let kHiddenDimension = 512


    //CREATE A FOLDER TO CONTAIN THE END-RESULT IMAGES OF THE OPTIMIZATION
    let str: String
    if useClassifier{
      str = "NNC"
    } else {
      str = "RAE"
    }
    // let folderName = "Results/ErrorValueVizualized_\(str)_\(kHiddenDimension)_\(featureSize)_5"
    let folderName = "Results/ErrorValueVizualized_\(str)Small_\(featureSize)_1"
    if !FileManager.default.fileExists(atPath: folderName) {
    do {
        try FileManager.default.createDirectory(atPath: folderName, withIntermediateDirectories: true, attributes: nil)
    } catch {
        print(error.localizedDescription)
    }
    }



    let firstGroundTruth = firstObb.center
    print("oBB coordinates", firstGroundTruth.t.x, firstGroundTruth.t.y)

    //CREATE A FIG
    print("hello1")
    let (fig, axs) = plt.subplots(1,2).tuple2
    let fr = np.squeeze(firstFrame.makeNumpyArray())
    axs[0].imshow(fr / 255.0, cmap: "gray")

        
    axs[0].set_xlim(firstGroundTruth.t.x - range/2, firstGroundTruth.t.x + range/2)
    axs[0].set_ylim(firstGroundTruth.t.y - range/2, firstGroundTruth.t.y + range/2)
    // axs[1].set_xlim(firstGroundTruth.t.x - range/2, firstGroundTruth.t.x + range/2)
    // axs[1].set_ylim(firstGroundTruth.t.y - range/2, firstGroundTruth.t.y + range/2)
    axs[1].set_xlim(0, range)
    axs[1].set_ylim(0, range)
    
    let x = firstGroundTruth.t.x
    let y = firstGroundTruth.t.y



    

    var values = Tensor<Double>(zeros: [Int(range), Int(range)])
    // var values = Tensor<Double>(zeros:firstFrame.shape)
    print("printing tensor",values)

    if useClassifier {
      // var classifier = NNClassifier(
      //   imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels, hiddenDimension: kHiddenDimension, latentDimension: featureSize
      // )
      var classifier = SmallerNNClassifier(
      imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels, latentDimension: featureSize
    )
      // classifier.load(weights: np.load("./classifiers/classifiers_today/classifier_weight_\(kHiddenDimension)_\(featureSize)_5.npy", allow_pickle: true))
      classifier.load(weights: np.load("./classifiers/classifiers_today/small_classifier_weight_\(featureSize)_1.npy", allow_pickle: true))

      print("done loading")
      for i in 0...Int(range)-1 {
        for j in 0...Int(range)-1 {
            let t = Vector2(x-range/2+Double(i), y-range/2+Double(j))
            // print("here3")
            let p = Pose2(firstGroundTruth.rot, t)
            var v = VariableAssignments()
            let poseId = v.store(p)
            let startpose = v[poseId]
            var fg = FactorGraph()
                // CREATE THE FACTOR AND FACTOR GRAPH
            let factorNNC = ProbablisticTrackingFactor2(poseId,
            measurement: firstFrame,
            classifier: classifier,
            patchSize: (40, 70),
            appearanceModelSize: (40, 70)
            )
            fg.store(factorNNC)
            // print("values at ij", values[i,j], factorNNC.errorVector(v[poseId]).x)
            // print("error vector", Tensor<Double>([factorNNC.errorVector(v[poseId]).x]))
            // print("value", (values[Int(x-range/2)+i,Int(y-range/2)+j]))
            // values[Int(x-range/2)+i,Int(y-range/2)+j] = Tensor<Double>([factorNNC.errorVector(v[poseId]).x])
            values[i,j] = Tensor<Double>(factorNNC.errorVector(v[poseId]).x)





        }
        print("row", i)
      }
      // print(values[0...,0])
      let min_val = values.min()
      if Double(min_val)! < 0 {
        values = values-min_val
      }
      values = values/values.max()*255
      print(values[0...,0])
      print(values.shape)
      axs[1].imshow(values.makeNumpyArray())


     
    //   axes.set_title(String(axes.get_title())! + "\n final err = \(final_err)" 
    //   + "\n label err = \(label_err).x)" 
    //   + "\n start err = \(start_err)"
    //   + "\n learning rate = \(lr)"
    //   + "\n converged = \(conv)")
    //   figs.savefig(folderName + "/optimization_final_\(j).png", bbox_inches: "tight")
    //   // let (figs2, axes2) = plotXYandTheta(xs: xs, ys: ys, thetas: thetas)
    //   // figs2.savefig(folderName + "/optimization_final_\(j)_XYtheta.png", bbox_inches: "tight")
    //   plt.close("all")
      fig.savefig(folderName + "/vizual_NNC.png", bbox_inches: "tight")

      

      
        

      



    } else {
        print("RAE")
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
        for i in 0...Int(range)-1 {
            for j in 0...Int(range)-1 {
                let t = Vector2(x-50.0+Double(i), y-50.0+Double(j))
                let p = Pose2(firstGroundTruth.rot, t)
                var v = VariableAssignments()
                let poseId = v.store(p)
                let startpose = v[poseId]
                var fg = FactorGraph()
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
                // print("values at ij", values[i,j], factorNNC.errorVector(v[poseId]).x)
                values[i,j] = Tensor<Double>(factorRAE.errorVector(v[poseId]).x)




            }
            print("row", i)
        }
        print(values[0...,0])
        let min_val = values.min()
        if Double(min_val)! < 0 {
            values = values-min_val
        }
        values = values/values.max()*255
        print(values[0...,0])
        print(values.shape)
        axs[1].imshow(values.makeNumpyArray())


     
    //   axes.set_title(String(axes.get_title())! + "\n final err = \(final_err)" 
    //   + "\n label err = \(label_err).x)" 
    //   + "\n start err = \(start_err)"
    //   + "\n learning rate = \(lr)"
    //   + "\n converged = \(conv)")
    //   figs.savefig(folderName + "/optimization_final_\(j).png", bbox_inches: "tight")
    //   // let (figs2, axes2) = plotXYandTheta(xs: xs, ys: ys, thetas: thetas)
    //   // figs2.savefig(folderName + "/optimization_final_\(j)_XYtheta.png", bbox_inches: "tight")
    //   plt.close("all")
      fig.savefig(folderName + "/vizual_RAE.png", bbox_inches: "tight")

      


      
    }
  }
}