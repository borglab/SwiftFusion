import ArgumentParser
import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation
import PenguinStructures

/// Brando11: compute the mean displacement
struct Brando11: ParsableCommand {
  @Option(help: "Run for number of frames")
  var trackLength: Int = 80

  func run() {
    let np = Python.import("numpy")
    let plt = Python.import("matplotlib.pyplot")
    let trainingDatasetSize = 100

    // LOAD THE IMAGE AND THE GROUND TRUTH ORIENTED BOUNDING BOX
    let dataDir = URL(fileURLWithPath: "./OIST_Data")
    let data = OISTBeeVideo(directory: dataDir, afterIndex: trainingDatasetSize, length: trackLength)!
    var dX = [Double]()
    var dY = [Double]()
    var dTheta = [Double]()
    for track in data.tracks  {
        var prevObb: OrientedBoundingBox?
        prevObb = nil
        for obb in track.boxes {
            if prevObb == nil {
                prevObb = obb
            } else {
                dX.append(obb.center.t.x - (prevObb)!.center.t.x)
                dY.append(obb.center.t.y - (prevObb)!.center.t.y)
                dTheta.append(obb.center.rot.theta - (prevObb)!.center.rot.theta)
            }
        }
    }
    // Plot histogram.

  }
}
