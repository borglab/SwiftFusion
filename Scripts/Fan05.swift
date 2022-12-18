import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import PythonKit
import Foundation
// import TensorFlow

/// Fan05: Error Landscape
struct Fan05: ParsableCommand {
  @Option(help: "Size of feature space")
  var featureSize: Int = 100

  @Option(help: "Which frame to show")
  var frameId: Int = 0

  @Option(help: "Which track to show")
  var trackId: Int = 0

  typealias LikelihoodModel = TrackingLikelihoodModel<PretrainedDenseRAE, MultivariateGaussian, MultivariateGaussian>

  func getTrainingDataEM(
    from dataset: OISTBeeVideo,
    numberForeground: Int = 300,
    numberBackground: Int = 300
  ) -> [LikelihoodModel.Datum] {
    let bgBoxes = dataset.makeBackgroundBoundingBoxes(patchSize: (40, 70), batchSize: numberBackground).map {
      (frame: $0.frame, type: LikelihoodModel.PatchType.bg, obb: $0.obb)
    }
    let fgBoxes = dataset.makeForegroundBoundingBoxes(patchSize: (40, 70), batchSize: numberForeground).map {
      (frame: $0.frame, type: LikelihoodModel.PatchType.fg, obb: $0.obb)
    }
    
    return fgBoxes + bgBoxes
  }
  
  // Visualize error landscape of PCA
  // Make sure you have a folder `Results/fan05` before running
  func run() {
    let dataDir = URL(fileURLWithPath: "./OIST_Data")

    let np = Python.import("numpy")

      // train foreground and background model and create tracker
    let trainingDataset = OISTBeeVideo(directory: dataDir, length: 100)!
    let trainingData = getTrainingDataEM(from: trainingDataset)

    let generator = ARC4RandomNumberGenerator(seed: 42)
    var em = MonteCarloEM<LikelihoodModel>(sourceOfEntropy: generator)

    let kHiddenDimension = 100
    let trackingModel = em.run(
      with: trainingData,
      iterationCount: 3,
      hook: { i, _, _ in
        print("EM run iteration \(i)")
      },
      given: LikelihoodModel.HyperParameters(
        encoder: PretrainedDenseRAE.HyperParameters(hiddenDimension: kHiddenDimension, latentDimension: featureSize, weightFile: "./oist_rae_weight_\(featureSize).npy")
      )
    )
    
    var statistics = FrameStatistics(Tensor([1.0]))
    statistics.mean = Tensor(0.0)
    statistics.standardDeviation = Tensor(1.0)

    let deltaXRange = Array(-60..<60).map { Double($0) }
    let deltaYRange = Array(-40..<40).map { Double($0) }

    let datasetToShow = OISTBeeVideo(directory: dataDir, afterIndex: frameId - 1, length: 2)!
    let frame = datasetToShow.frames[1]
    let pose = datasetToShow.tracks[trackId].boxes[0].center
    let (fig, _) = plotErrorPlaneTranslation(
      frame: frame,
      at: pose,
      deltaXs: deltaXRange,
      deltaYs: deltaYRange,
      statistics: statistics,
      encoder: trackingModel.encoder,
      foregroundModel: trackingModel.foregroundModel,
      backgroundModel: trackingModel.backgroundModel
    )
    fig.savefig("Results/fan05/fan05_em_ae_mg_mg_\(trackId)_\(frameId)_\(featureSize).pdf", bbox_inches: "tight")
    fig.savefig("Results/fan05/fan05_em_ae_mg_mg_\(trackId)_\(frameId)_\(featureSize).png", bbox_inches: "tight")
  }
}
