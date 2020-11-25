import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation

/// Fan09: Raw Tracker, with Identity projection
struct Fan09: ParsableCommand {
  @Option(help: "Run on track number x")
  var trackId: Int = 0
  
  @Option(help: "Run for number of frames")
  var trackLength: Int = 80
  
  @Option(help: "Size of feature space")
  var featureSize: Int = 30

  // Just runs an RP tracker and saves image to file
  // Make sure you have a folder `Results/frank02` before running
  func run() {
    let np = Python.import("numpy")
    let dataDir = URL(fileURLWithPath: "./OIST_Data")

    let (imageHeight, imageWidth, imageChannels) = (40, 70, 1)

    let mean = Tensor<Double>(numpy: np.load("./pca_mu_\(featureSize).npy"))!.reshaped(to: [imageHeight, imageWidth, imageChannels])
    let encoder = IdentityProjection(fromShape: TensorShape([imageHeight, imageWidth, imageChannels]), sampleMean: mean)

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
