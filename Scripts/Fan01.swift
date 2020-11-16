import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import PythonKit
import Foundation

/// Fan01: AE Tracker, with sampling-based initialization
struct Fan01: ParsableCommand {
  @Option(help: "Run on track number x")
  var trackId: Int = 0
  
  @Option(help: "Run for number of frames")
  var trackLength: Int = 80
  
  @Option(help: "Size of feature space")
  var featureSize: Int = 30

  @Option(help: "Pretrained weights")
  var weightsFile: String = "./oist_rae_weight_30.npy"

  // Just runs an RP tracker and saves image to file
  // Make sure you have a folder `Results/frank02` before running
  func run() {
    let np = Python.import("numpy")
    let kHiddenDimension = 100

    let dataDir = URL(fileURLWithPath: "./OIST_Data")
    let (imageHeight, imageWidth, imageChannels) =
      (40, 70, 1)
    var rae = DenseRAE(
      imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels,
      hiddenDimension: kHiddenDimension, latentDimension: featureSize
    )
    rae.load(weights: np.load(weightsFile, allow_pickle: true))

    let (fig, _, _) = runProbabilisticTracker(
      directory: dataDir,
      encoder: rae,
      onTrack: trackId, forFrames: trackLength, withSampling: true,
      withFeatureSize: featureSize,
      savePatchesIn: "Results/fan01"
    )

    /// Actual track v.s. ground truth track
    fig.savefig("Results/fan01/fan01_track\(trackId)_\(featureSize).pdf", bbox_inches: "tight")
  }
}
