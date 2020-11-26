import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation

/// Fan09: PCA Tracker, with 2-cluster Gaussian Mixture
struct Fan09: ParsableCommand {
  @Option(help: "Run on track number x")
  var trackId: Int = 0
  
  @Option(help: "Run for number of frames")
  var trackLength: Int = 80
  
  @Option(help: "Size of feature space")
  var featureSize: Int = 30

  // Make sure you have a folder `Results/fan09` before running
  func run() {
    let np = Python.import("numpy")
    let dataDir = URL(fileURLWithPath: "./OIST_Data")

    let encoder = PCAEncoder(
      withBasis: Tensor<Double>(numpy: np.load("./pca_U_\(featureSize).npy"))!,
      andMean: Tensor<Double>(numpy: np.load("./pca_mu_\(featureSize).npy"))!
    )

    let (fig, track, gt) = runProbabilisticTracker(
      directory: dataDir,
      encoder: encoder,
      onTrack: trackId, forFrames: trackLength, withSampling: true,
      withFeatureSize: featureSize,
      savePatchesIn: "Results/fan09"
    )

    /// Actual track v.s. ground truth track
    fig.savefig("Results/fan09/fan09_track\(trackId)_\(featureSize).pdf", bbox_inches: "tight")
    fig.savefig("Results/fan09/fan09_track\(trackId)_\(featureSize).png", bbox_inches: "tight")

    let json = JSONEncoder()
    json.outputFormatting = .prettyPrinted

    let track_data = try! json.encode(track)
    try! track_data.write(to: URL(fileURLWithPath: "Results/fan09/fan09_track_\(trackId)_\(featureSize).json"))

    let gt_data = try! json.encode(gt)
    try! gt_data.write(to: URL(fileURLWithPath: "Results/fan09/fan09_gt_\(trackId)_\(featureSize).json"))
  }
}
