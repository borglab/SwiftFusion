import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation

/// Fan03: RP Tracker, with sampling-based initialization
struct Fan03: ParsableCommand {
  @Option(help: "Run on track number x")
  var trackId: Int = 0
  
  @Option(help: "Run for number of frames")
  var trackLength: Int = 80
  
  @Option(help: "Size of feature space")
  var featureSize: Int = 100

  // Just runs an RP tracker and saves image to file
  // Make sure you have a folder `Results/frank02` before running
  func run() {
    let dataDir = URL(fileURLWithPath: "./OIST_Data")
    let (imageHeight, imageWidth, imageChannels) =
      (40, 70, 1)
    
    let rp = RandomProjection(fromShape: TensorShape([imageHeight, imageWidth, imageChannels]), toFeatureSize: featureSize)

    let (fig, _, _) = runProbabilisticTracker(
      directory: dataDir,
      encoder: rp,
      onTrack: trackId, forFrames: trackLength, withSampling: true,
      withFeatureSize: featureSize,
      savePatchesIn: "Results/fan03"
    )

    /// Actual track v.s. ground truth track
    fig.savefig("Results/fan03/fan03_track\(trackId)_\(featureSize).pdf", bbox_inches: "tight")
  }
}
