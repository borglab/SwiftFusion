import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
// import TensorFlow
import PythonKit
import Foundation

/// Fan06: RAE Tracker, using the new Tracking
struct Fan06: ParsableCommand {
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
    let kHiddenDimension = 100
    let np = Python.import("numpy")
    let dataDir = URL(fileURLWithPath: "./OIST_Data")

    let (imageHeight, imageWidth, imageChannels) =
      (40, 70, 1)
    var rae = DenseRAE(
      imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels,
      hiddenDimension: kHiddenDimension, latentDimension: featureSize
    )

    rae.load(weights: np.load("./oist_rae_weight_\(featureSize).npy", allow_pickle: true))

    let (fig, track, gt) = runProbabilisticTracker(
      directory: dataDir,
      encoder: rae,
      onTrack: trackId, forFrames: trackLength, withSampling: true,
      withFeatureSize: featureSize,
      savePatchesIn: "Results/fan06"
    )

    /// Actual track v.s. ground truth track
    fig.savefig("Results/fan06/fan06_track\(trackId)_\(featureSize).pdf", bbox_inches: "tight")
    fig.savefig("Results/fan06/fan06_track\(trackId)_\(featureSize).png", bbox_inches: "tight")

    let json = JSONEncoder()
    json.outputFormatting = .prettyPrinted

    let track_data = try! json.encode(track)
    try! track_data.write(to: URL(fileURLWithPath: "Results/fan06/fan06_track_\(trackId)_\(featureSize).json"))

    let gt_data = try! json.encode(gt)
    try! gt_data.write(to: URL(fileURLWithPath: "Results/fan06/fan06_gt_\(trackId)_\(featureSize).json"))
  }
}
