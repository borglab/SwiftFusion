import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import PythonKit
import Foundation

/// Frank01: Random Projections Tracker, with sampling-based initialization
struct Frank02: ParsableCommand {
  @Option(help: "Run on track number x")
  var trackId: Int = 0
  
  @Option(help: "Run for number of frames")
  var trackLength: Int = 80
  
  @Option(help: "Size of feature space")
  var featureSize: Int = 100

  // Just runs an RP tracker and saves image to file
  // Make sure you have a folder `Results/frank02` before running
  // TODO: use the generic version instead, remove runRPTracker
  func run() {
    let (fig, _, _) = runRPTracker(
      directory: URL(fileURLWithPath: "./OIST_Data"),
      onTrack: trackId, forFrames: trackLength, withSampling: true,
      withFeatureSize: featureSize,
      savePatchesIn: "Results/frank02"
    )

    /// Actual track v.s. ground truth track
    fig.savefig("Results/frank02/frank02_track\(trackId)_\(featureSize).pdf", bbox_inches: "tight")
  }
}
