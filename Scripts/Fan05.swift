import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import PythonKit
import Foundation
import TensorFlow

/// Fan05: Error Landscape
struct Fan05: ParsableCommand {
  @Option(help: "Size of feature space")
  var featureSize: Int = 100

  @Option(help: "Which frame to show")
  var frameId: Int = 0

  @Option(help: "Which track to show")
  var trackId: Int = 0

  // Visualize error landscape of PCA
  // Make sure you have a folder `Results/fan05` before running
  func run() {
    let dataDir = URL(fileURLWithPath: "./OIST_Data")

    let np = Python.import("numpy")
    
      // train foreground and background model and create tracker
    let trainingData = OISTBeeVideo(directory: dataDir, length: 100)!
    // let testData = OISTBeeVideo(directory: dataDir, afterIndex: 100, length: forFrames)!

    let (imageHeight, imageWidth, imageChannels) = (40, 70, 1)
    // let encoder = RandomProjection(fromShape: TensorShape([imageHeight, imageWidth, imageChannels]), toFeatureSize: featureSize)
  
    // let encoder = PCAEncoder(
    //   withBasis: Tensor<Double>(numpy: np.load("./pca_U_\(featureSize).npy"))!,
    //   andMean: Tensor<Double>(numpy: np.load("./pca_mu_\(featureSize).npy"))!
    // )

    var encoder = DenseRAE(
      imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels,
      hiddenDimension: 100, latentDimension: featureSize
    )

    encoder.load(weights: np.load("./oist_rae_weight_\(featureSize).npy", allow_pickle: true))

    let (fg, bg, statistics) = getTrainingBatches(
      dataset: trainingData, boundingBoxSize: (40, 70),
      fgBatchSize: 3000,
      bgBatchSize: 3000,
      fgRandomFrameCount: 100,
      bgRandomFrameCount: 100,
      useCache: true
    )

    let batchPositive = encoder.encode(fg)
    let foregroundModel = MultivariateGaussian(from: batchPositive, regularizer: 1e-3)

    let batchNegative = encoder.encode(bg)
    let backgroundModel = MultivariateGaussian(from: batchNegative, regularizer: 1e-3)

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
      encoder: encoder,
      foregroundModel: foregroundModel,
      backgroundModel: backgroundModel
    )
    fig.savefig("Results/fan05/fan05_pf_ae_mg_mg_\(trackId)_\(frameId)_\(featureSize).pdf", bbox_inches: "tight")
    fig.savefig("Results/fan05/fan05_pf_ae_mg_mg_\(trackId)_\(frameId)_\(featureSize).png", bbox_inches: "tight")
  }
}
