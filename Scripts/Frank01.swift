import ArgumentParser

import SwiftFusion
import BeeTracking
import PythonKit
import Foundation

/// Frank01: Random Projections Baseline Tracker
struct Frank01: ParsableCommand {

  // Just runs an RP tracker and saves image to file
  func run() {
    let fig: PythonObject = runRPTracker(directory: URL(fileURLWithPath: "./OIST_Data"), onTrack: 15)
    fig.savefig("frank01.pdf", bbox_inches: "tight")
  }
}
