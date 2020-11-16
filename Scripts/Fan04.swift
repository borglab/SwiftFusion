import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation

/// Fan04: PPCA Tracker, with sampling-based initialization
struct Fan04: ParsableCommand {
  @Option(help: "Run on track number x")
  var trackId: Int = 0
  
  @Option(help: "Run for number of frames")
  var trackLength: Int = 80
  
  @Option(help: "Size of feature space")
  var featureSize: Int = 30

  @Flag(help: "Training mode")
  var training: Bool = false

  // Just runs an RP tracker and saves image to file
  // Make sure you have a folder `Results/frank02` before running
  func run() {
    let np = Python.import("numpy")
    let dataDir = URL(fileURLWithPath: "./OIST_Data")
    let (imageHeight, imageWidth) =
      (40, 70)

    var ppca = PPCA(latentSize: featureSize)
    if training {
      let ppcaTrainingData = OISTBeeVideo(directory: dataDir, length: 10)!
      var statistics = FrameStatistics(Tensor<Double>(0.0))
      statistics.mean = Tensor(62.26806976644069)
      statistics.standardDeviation = Tensor(37.44683834503672)
      let trainingBatch = ppcaTrainingData.makeBatch(statistics: statistics, appearanceModelSize: (imageHeight, imageWidth))
      ppca.train(images: trainingBatch)
      np.save("./ppca_W_\(featureSize)", ppca.W.makeNumpyArray())
      np.save("./ppca_mu_\(featureSize)", ppca.mu.makeNumpyArray())
    } else {
      ppca.W = Tensor<Double>(numpy: np.load("./ppca_W_\(featureSize).npy"))!
      ppca.mu = Tensor<Double>(numpy: np.load("./ppca_mu_\(featureSize).npy"))!
      let W_m = ppca.W.reshaped(to: [imageHeight * imageWidth * 1, featureSize])
      ppca.W_inv = matmul(pinv(matmul(W_m.transposed(), W_m)), W_m.transposed())
    }

    let (fig, _, _) = runProbabilisticTracker(
      directory: dataDir,
      encoder: ppca,
      onTrack: trackId, forFrames: trackLength, withSampling: true,
      withFeatureSize: featureSize,
      savePatchesIn: "Results/fan04"
    )

    /// Actual track v.s. ground truth track
    fig.savefig("Results/fan04/fan04_track\(trackId)_\(featureSize).pdf", bbox_inches: "tight")
  }
}
