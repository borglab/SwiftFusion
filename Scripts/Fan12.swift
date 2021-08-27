import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation

/// Fan12: RAE training
struct Fan12: ParsableCommand {
  typealias LikelihoodModel = TrackingLikelihoodModel<PretrainedDenseRAE, MultivariateGaussian, GaussianNB>

  @Option(help: "Size of feature space")
  var featureSize: Int = 5

  @Flag(help: "Training mode")
  var training: Bool = false

  func getTrainingData(
    from dataset: OISTBeeVideo,
    numberForeground: Int = 3000
  ) -> [LikelihoodModel.Datum] {
    let fgBoxes = dataset.makeForegroundBoundingBoxes(patchSize: (40, 70), batchSize: numberForeground).map {
      (frame: $0.frame, type: LikelihoodModel.PatchType.fg, obb: $0.obb)
    }
    
    return fgBoxes
  }
  
  // Just runs an RP tracker and saves image to file
  // Make sure you have a folder `Results/fan12` before running
  func run() {
    let kHiddenDimension = 512
    let dataDir = URL(fileURLWithPath: "./OIST_Data")

    let trainingDataset = OISTBeeVideo(directory: dataDir, length: 100)!
    
    let trainingData = Tensor<Double>(stacking: getTrainingData(from: trainingDataset).map { $0.frame!.patch(at: $0.obb) })
    
    print("Training...")
    let rae: PretrainedDenseRAE = PretrainedDenseRAE(
      trainFrom: trainingData,
      given: PretrainedDenseRAE.HyperParameters(hiddenDimension: kHiddenDimension, latentDimension: featureSize, weightFile: "")
    )
    
    rae.save(to: "./oist_rae_weight_\(featureSize).npy")
  }
}
