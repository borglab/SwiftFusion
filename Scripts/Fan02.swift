import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import PythonKit
import Foundation
import TensorFlow

/// Fan02: Random Projections Error Landscape
struct Fan02: ParsableCommand {
  @Option(help: "Size of feature space")
  var featureSize: Int = 100

  // Visualize error landscape of PCA
  // Make sure you have a folder `Results/fan02` before running
  func run() {
    let dataDir = URL(fileURLWithPath: "./OIST_Data")

    let np = Python.import("numpy")
    let plt = Python.import("matplotlib.pyplot")
    
      // train foreground and background model and create tracker
    let trainingData = OISTBeeVideo(directory: dataDir, length: 100)!
    // let testData = OISTBeeVideo(directory: dataDir, afterIndex: 100, length: forFrames)!

    // let (imageHeight, imageWidth, imageChannels) = (40, 70, 1)
    // let encoder = RandomProjection(fromShape: TensorShape([imageHeight, imageWidth, imageChannels]), toFeatureSize: featureSize)
    let encoder = PCAEncoder(withBasis: Tensor<Double>(numpy: np.load("./pca_U_\(featureSize).npy"))!)

    let (fg, _, _) = getTrainingBatches(
      dataset: trainingData, boundingBoxSize: (40, 70),
      fgBatchSize: 3000,
      bgBatchSize: 3000,
      fgRandomFrameCount: 100,
      bgRandomFrameCount: 100,
      useCache: true
    )

    let batchPositive = encoder.encode(fg)
//    let foregroundModel = MultivariateGaussian(from:batchPositive, regularizer: 1e-3)
//
//    let batchNegative = encoder.encode(bg)
//    let backgroundModel = GaussianNB(from: batchNegative, regularizer: 1e-3)

    // print(foregroundModel.covariance_inv!.diagonalPart())
    // print(backgroundModel.sigmas!)

    // for i in 0..<min(30, encoder.d) {
    //   let (fig, ax) = plt.subplots(1, 1).tuple2
    //   let dots = TensorRange.ellipsis
    //   let u_plot = ax.imshow(encoder.U[dots, i].reshaped(to: [40, 70, 1]).makeNumpyArray())
    //   fig.colorbar(u_plot, ax: ax)
    //   fig.savefig("Results/fan02/fan02_U_\(i)_\(featureSize).pdf", bbox_inches: "tight")
    // }

    let (fig, ax) = plt.subplots(1, 2).tuple2
    let original = fg[0]
    let reconstructed = matmul(
      encoder.U.reshaped(to: [40*70, featureSize]), batchPositive[0].expandingShape(at: 1)
    ).reshaped(to: [40, 70, 1])

    ax[0].imshow(original.makeNumpyArray())
    ax[1].imshow(reconstructed.makeNumpyArray())
    fig.savefig("Results/fan02/fan02_recon_\(featureSize).pdf", bbox_inches: "tight")
  }
}
