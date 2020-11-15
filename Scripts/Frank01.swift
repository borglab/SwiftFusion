import ArgumentParser

import SwiftFusion
import BeeTracking
import PythonKit

/// Frank01: Random Projections Baseline Tracker
struct Frank01: ParsableCommand {

  func run() {
    let fig: PythonObject = runRPTracker(onTrack: 15)

    fig.savefig("frank01.pdf", bbox_inches: "tight")
  }
}