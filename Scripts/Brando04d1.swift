import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation




/// Brando04: NNClassifier training
struct Brando04d1: ParsableCommand {
  typealias LikelihoodModel = TrackingLikelihoodModel<PretrainedDenseRAE, MultivariateGaussian, GaussianNB>


  @Flag(help: "Training mode")
  var training: Bool = false

  let num_boxes: Int = 10000
  let pert = Vector3(0.0, 30, 0)

  func getTrainingDataBG(
    from dataset: OISTBeeVideo
  ) -> (Tensor<Float>, Tensor<Double>) {
    print("bg")
    let frames_obbs = dataset.makeBackgroundBoundingBoxes(patchSize: (40, 70), batchSize: num_boxes)
    var bgBoxes = [Tensor<Double>]()
    for i in 0...frames_obbs.count-1 {
      var obb = frames_obbs[i].obb
      obb.center.perturbWith(stddev: pert)
      bgBoxes.append(frames_obbs[i].frame!.patch(at: obb))
    
    }
    
    print("bg2")
    let labels = Tensor<Float>(ones: [num_boxes])
    print("labels done bg")
    let patches = Tensor<Double>(stacking: bgBoxes.map {$0})
    print("patches done bg")
    return (labels, patches)
  }
  


  func getTrainingDataFG(
    from dataset: OISTBeeVideo
  ) -> (Tensor<Float>, Tensor<Double>) {
    print("fg")
    let frames_obbs = dataset.makeForegroundBoundingBoxes(patchSize: (40, 70), batchSize: num_boxes)
    var fgBoxes = [Tensor<Double>]()
    for i in 0...frames_obbs.count-1 {
      var obb = frames_obbs[i].obb
      obb.center.perturbWith(stddev: pert)
      fgBoxes.append(frames_obbs[i].frame!.patch(at: obb))
    
    }
    
    print("bg2")
    let labels = Tensor<Float>(ones: [num_boxes])
    print("labels done bg")
    let patches = Tensor<Double>(stacking: fgBoxes.map {$0})
    print("patches done bg")
    return (labels, patches)
  }



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
    let iterations = [1]

    
    let lr = Float(1e-6)
    for i in iterations {
      let pretrained_weights = "./classifiers/classifiers_today/classifier_weight_\(kHiddenDimension)_\(featSize)_\(i)_20000boxes_300epochs_retrained(0.0, 30, 0)_lr=\(lr).npy"
      let path = "./classifiers/classifiers_today/classifier_weight_\(kHiddenDimension)_\(featSize)_\(i)_20000boxes_300epochs_retrained(0.0, 30, 0)_lr=\(lr)_2nd_iter.npy"
      if FileManager.default.fileExists(atPath: path) {
          print("File Already Exists. Abort training")
          continue
      }
      print("Training...")
      let rae: PretrainedNNClassifier = PretrainedNNClassifier(
        patches: patches,
        labels: labels,
        given: PretrainedNNClassifier.HyperParameters(hiddenDimension: kHiddenDimension, latentDimension: featSize, weightFile: pretrained_weights, learningRate: lr),
        train_mode: "pretrained"
      )
      rae.save(to: path)

    }

    
    
  }
}
