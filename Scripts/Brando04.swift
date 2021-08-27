import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation




/// Fan12: RAE training
struct Brando04: ParsableCommand {
  typealias LikelihoodModel = TrackingLikelihoodModel<PretrainedDenseRAE, MultivariateGaussian, GaussianNB>

  

  @Flag(help: "Training mode")
  var training: Bool = false

  let num_boxes: Int = 3000

  func getTrainingDataBG(
    from dataset: OISTBeeVideo,
    numberForeground: Int = 3000
  ) -> [LikelihoodModel.Datum] {
    print("bg")

    // var allBoxes = [LikelihoodModel.Datum]()
    let bgBoxes = dataset.makeBackgroundBoundingBoxes(patchSize: (40, 70), batchSize: num_boxes).map {
      (frame: $0.frame, type: LikelihoodModel.PatchType.bg, obb: $0.obb)
    }
    print("bg2")


    return bgBoxes
  }

  func getTrainingDataFG(
    from dataset: OISTBeeVideo,
    numberForeground: Int = 3000
  ) -> [LikelihoodModel.Datum] {
    print("fg")
    // var allBoxes = [LikelihoodModel.Datum]()
    let fgBoxes = dataset.makeForegroundBoundingBoxes(patchSize: (40, 70), batchSize: num_boxes).map {
      (frame: $0.frame, type: LikelihoodModel.PatchType.fg, obb: $0.obb)
    }
    print("fg2")

    return fgBoxes
  }




  // Just runs an RP tracker and saves image to file
  // Make sure you have a folder `Results/fan12` before running
  func run() {
    let folderName = "classifiers/classifiers_today"
    if !FileManager.default.fileExists(atPath: folderName) {
      do {
          try FileManager.default.createDirectory(atPath: folderName, withIntermediateDirectories: true, attributes: nil)
      } catch {
          print(error.localizedDescription)
      }
    } else {
      print("folder exists")
    }
    
    
    let dataDir = URL(fileURLWithPath: "./OIST_Data")
    print("hello")

    // if I call makeBackgroundBoundingBoxes, makeForegroundBoundingBoxes.
    let trainingDataset = OISTBeeVideo(directory: dataDir, length: 100)!
    print("done")
    var bgBoxes = getTrainingDataBG(from: trainingDataset)
    print(bgBoxes.count)
    // let trainingDataset2 = OISTBeeVideo(directory: dataDir, length: 100)!
    print("2")
    var fgBoxes = getTrainingDataFG(from: trainingDataset)
    print(fgBoxes.count)
    
    // print("all boxes")
    var allBoxes = [LikelihoodModel.Datum]()
    for i in 0...(fgBoxes.count-1)/100 {
      //appending 100 bounding boxes
      for j in 0...99 {
        allBoxes.append(bgBoxes[j+i*100])
      }
      //appending 100 bounding boxes
      for j in 0...99 {
        allBoxes.append(fgBoxes[j+i*100])
      }
    }
    print("total boxes", allBoxes.count)
    // for i in 0...allBoxes.count-1 {
    //   print(i)
    //   print(allBoxes[i].type)
    //   print(allBoxes[i].obb)
    // }



    let patches = Tensor<Double>(stacking: allBoxes.map { $0.frame!.patch(at: $0.obb)})
    let labels = Tensor<Int32>(stacking: allBoxes.map { $0.type == TrackingLikelihoodModel<PretrainedDenseRAE, MultivariateGaussian, GaussianNB>.PatchType.bg ? Tensor<Int32>(0) : Tensor<Int32>(1)})
    print("shape of patches", patches.shape)
    print("shape of labels", labels.shape)
    // return

    // let trainingData = allBoxes
    // let trainingData = (images, labels)
    // print("training data shape", trainingData.shape)
    print("training data done")
    // for featSize in [64,128,256] {
    // for kHiddenDimension in [256,512] {
    let kHiddenDimension = 512
    let featSize = 256
    for i in 1...7 {
      print("Training...")
      // let rae: PretrainedNNClassifier = PretrainedNNClassifier(
      //   patches: patches,
      //   labels: labels,
      //   given: PretrainedNNClassifier.HyperParameters(hiddenDimension: kHiddenDimension, latentDimension: featSize, weightFile: "")
      // )
      // rae.save(to: "./classifiers/classifiers_today/classifier_weight_\(kHiddenDimension)_\(featSize)_\(i).npy")
      // let rae: PretrainedSmallerNNClassifier = PretrainedSmallerNNClassifier(
      //   patches: patches,
      //   labels: labels,
      //   given: PretrainedSmallerNNClassifier.HyperParameters(latentDimension: featSize, weightFile: "")
      // )
      // rae.save(to: "./classifiers/classifiers_today/small_classifier_weight_\(featSize)_\(i).npy")
      let rae: PretrainedLargerNNClassifier = PretrainedLargerNNClassifier(
        patches: patches,
        labels: labels,
        given: PretrainedLargerNNClassifier.HyperParameters(hiddenDimension: kHiddenDimension, latentDimension: featSize, weightFile: "")
      )
      rae.save(to: "./classifiers/classifiers_today/large_classifier_weight_\(kHiddenDimension)_\(featSize)_\(i).npy")
      print("saved")
    }
    // }
    // }
    
    
  }
}
