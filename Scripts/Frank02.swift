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
  func run() {
    let (fig, track, gt) = runRPTracker(
      directory: URL(fileURLWithPath: "./OIST_Data"),
      onTrack: trackId, forFrames: trackLength, withSampling: true,
      withFeatureSize: featureSize
    )

    /// Actual track v.s. ground truth track
    fig.savefig("Results/frank02/frank02_track\(trackId)_\(featureSize).pdf", bbox_inches: "tight")

    let testData = OISTBeeVideo(directory: URL(fileURLWithPath: "./OIST_Data"), afterIndex: 100, length: trackLength)!

    /// Plot all the frames so we can visually inspect the situation
    for i in track.indices {
      let (fig_initial, _) = plotPatchWithGT(frame: testData.frames[i], actual: track[i], expected: gt[i])
      fig_initial.savefig("Results/frank02/frank02_1st_img_track\(trackId)_\(featureSize)_\(i).png", bbox_inches: "tight")
    }
  }
}
