import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation




/// Brando15 SAVE PATCHES FOR LATER USE
struct Brando15: ParsableCommand {
  typealias LikelihoodModel = TrackingLikelihoodModel<PretrainedDenseRAE, MultivariateGaussian, GaussianNB>


  @Flag(help: "Training mode")
  var training: Bool = false

  let num_boxes: Int = 10000

  func getTrainingDataBG(
    from dataset: OISTBeeVideo
  ) -> (Tensor<Float>, Tensor<Double>) {
    print("bg")

    // var allBoxes = [LikelihoodModel.Datum]()
    let bgBoxes = dataset.makeBackgroundBoundingBoxes(patchSize: (40, 70), batchSize: num_boxes).map {
      // (frame: $0.frame, type: LikelihoodModel.PatchType.bg, obb: $0.obb)
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
    // var allBoxes = [LikelihoodModel.Datum]()
    let fgBoxes = dataset.makeForegroundBoundingBoxes(patchSize: (40, 70), batchSize: num_boxes).map {
      // (frame: $0.frame, type: LikelihoodModel.PatchType.bg, obb: $0.obb)
      $0.frame!.patch(at: $0.obb)
    }
    print("bg2")
    let labels = Tensor<Float>(ones: [num_boxes])
    print("labels done bg")
    let patches = Tensor<Double>(stacking: fgBoxes.map {$0})
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
    var (labels_bg, patches_bg) = getTrainingDataBG(from: trainingDataset)
    let np = Python.import("numpy")
    np.save("Patches_bg_\(num_boxes).npy", patches_bg.makeNumpyArray())
    var (labels_fg, patches_fg) = getTrainingDataFG(from: trainingDataset)

    // var patches = concatenate(patches_bg, patches_fg)
    var patches = Tensor(stacking: patches_bg.unstacked() + patches_fg.unstacked())
    var labels = Tensor<Int8>(concatenate(labels_bg, labels_fg))
    print("shape of patches", patches.shape)
    print("shape of labels", labels.shape)
    np.save("Patches_bg_fg_\(num_boxes).npy", patches.makeNumpyArray())
  }
}
