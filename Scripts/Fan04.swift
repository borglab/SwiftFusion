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
    let (imageHeight, imageWidth) = (40, 70)

    // var pca = RandomProjection(fromShape: TensorShape([imageHeight, imageWidth, 1]), toFeatureSize: featureSize)
    let pca: PCAEncoder = { () -> PCAEncoder in
      /// Training mode, trains the PPCA model and save to cache file
      if training {
        let pcaTrainingData = OISTBeeVideo(directory: dataDir, length: 100)!
        var statistics = FrameStatistics(Tensor<Double>(0.0))
        statistics.mean = Tensor(62.26806976644069)
        statistics.standardDeviation = Tensor(37.44683834503672)
        let trainingBatch = pcaTrainingData.makeBatch(statistics: statistics, appearanceModelSize: (imageHeight, imageWidth), batchSize: 3000)
        let encoder = PCAEncoder(from: trainingBatch, given: featureSize)
        np.save("./pca_U_\(featureSize)", encoder.U.makeNumpyArray())
        np.save("./pca_mu_\(featureSize)", encoder.mu.makeNumpyArray())
        return encoder
      } else {
        /// Just load the cached weight matrix
        return PCAEncoder(
          withBasis: Tensor<Double>(numpy: np.load("./pca_U_\(featureSize).npy"))!,
          andMean: Tensor<Double>(numpy: np.load("./pca_mu_\(featureSize).npy"))!
        )
      }
    }()

    let (fig, track, gt) = runProbabilisticTracker(
      directory: dataDir,
      encoder: pca,
      onTrack: trackId, forFrames: trackLength, withSampling: true,
      withFeatureSize: featureSize,
      savePatchesIn: "Results/fan04"
    )

    /// Actual track v.s. ground truth track
    fig.savefig("Results/fan04/fan04_track\(trackId)_\(featureSize).pdf", bbox_inches: "tight")
    fig.savefig("Results/fan04/fan04_track\(trackId)_\(featureSize).png", bbox_inches: "tight")

    let json = JSONEncoder()
    json.outputFormatting = .prettyPrinted

    let track_data = try! json.encode(track)
    try! track_data.write(to: URL(fileURLWithPath: "Results/fan04/fan04_track_\(trackId)_\(featureSize).json"))

    let gt_data = try! json.encode(gt)
    try! gt_data.write(to: URL(fileURLWithPath: "Results/fan04/fan04_gt_\(trackId)_\(featureSize).json"))
  }
}
