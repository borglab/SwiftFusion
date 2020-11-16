import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import PythonKit
import Foundation

/// Fan02: Random Projections Error Landscape
struct Fan02: ParsableCommand {
  @Option(help: "Run on track number x")
  var trackId: Int = 0
  
  @Option(help: "Run for number of frames")
  var trackLength: Int = 80
  
  @Option(help: "Size of feature space")
  var featureSize: Int = 100

  // Just runs an RP tracker and saves image to file
  // Make sure you have a folder `Results/fan02` before running
  func run() {
    let (fig, _, _) = runRPTracker(
      directory: URL(fileURLWithPath: "./OIST_Data"),
      onTrack: trackId, forFrames: trackLength, withSampling: true,
      withFeatureSize: featureSize,
      savePatchesIn: "Results/fan02"
    )

    /// Actual track v.s. ground truth track
    fig.savefig("Results/fan02/fan02_track\(trackId)_\(featureSize).pdf", bbox_inches: "tight")
  }
}
