import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation




/// Brando04: NNClassifier training
struct Brando04: ParsableCommand {
  typealias LikelihoodModel = TrackingLikelihoodModel<PretrainedDenseRAE, MultivariateGaussian, GaussianNB>


  @Flag(help: "Training mode")
  var training: Bool = false

  let num_boxes: Int = 10000

  func getTrainingDataBG(
    from dataset: OISTBeeVideo
  ) -> (Tensor<Float>, Tensor<Double>) {
    print("bg")

    let bgBoxes = dataset.makeBackgroundBoundingBoxes(patchSize: (40, 70), batchSize: num_boxes).map {
      $0.frame!.patch(at: $0.obb)
    }
    print("bg2")
    let labels = Tensor<Float>(zeros: [num_boxes])
    print("labels done bg")
    let patches = Tensor<Double>(stacking: bgBoxes.map {$0})
    print("patches done bg")
    return (labels, patches)
  }
  


  func getTrainingDataFG(
    from dataset: OISTBeeVideo
  ) -> (Tensor<Float>, Tensor<Double>) {
    print("fg")
    let bgBoxes = dataset.makeForegroundBoundingBoxes(patchSize: (40, 70), batchSize: num_boxes).map {
      $0.frame!.patch(at: $0.obb)
    }
    print("bg2")
    let labels = Tensor<Float>(zeros: [num_boxes])
    print("labels done bg")
    let patches = Tensor<Double>(stacking: bgBoxes.map {$0})
    print("patches done bg")
    return (labels, patches)
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
    let trainingDataset = OISTBeeVideo(directory: dataDir, length: 100)!
    var (labels_fg, patches_fg) = getTrainingDataFG(from: trainingDataset)
    var (labels_bg, patches_bg) = getTrainingDataBG(from: trainingDataset)
    

    var patches = Tensor(stacking: patches_bg.unstacked() + patches_fg.unstacked())
    var labels = Tensor<Int8>(concatenate(labels_bg, labels_fg))
    print("shape of patches", patches.shape)
    print("shape of labels", labels.shape)

    let kHiddenDimension = 512
    let featSize = 512
    let iterations = [5,6,7]

    for i in iterations {
      let path = "./classifiers/classifiers_today/classifier_weight_\(kHiddenDimension)_\(featSize)_\(i)_60000boxes_600epochs.npy"
      if FileManager.default.fileExists(atPath: path) {
          print("File Already Exists. Abort training")
          continue
      }
      print("Training...")
      let rae: PretrainedNNClassifier = PretrainedNNClassifier(
        patches: patches,
        labels: labels,
        given: PretrainedNNClassifier.HyperParameters(hiddenDimension: kHiddenDimension, latentDimension: featSize, weightFile: "", learningRate: 1e-3),
        train_mode: "from_scratch"
      )
      rae.save(to: path)

    }

    
    
  }
}
