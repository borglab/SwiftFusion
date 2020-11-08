import ArgumentParser
import BeeDataset
import BeeTracking
import PenguinParallelWithFoundation
import PythonKit
import SwiftFusion
import TensorFlow
import Foundation

struct OISTVisualizationTool: ParsableCommand {
  static var configuration = CommandConfiguration(
    subcommands: [ViewFrame.self])
}

struct ViewFrame: ParsableCommand {
  @Option(help: "Location of dataset folder")
  var datasetLocation: String = "./OIST_Data"

  @Option(help: "Which frame to show")
  var frameId: Int = 0

  func run() {
    let dataURL = URL(fileURLWithPath: datasetLocation)
    print("Videwing \(dataURL) at frame \(frameId)")
    let dataset = OISTBeeVideo(directory: dataURL, deferLoadingFrames: true)!

    print(dataset.labels)
  }
}

OISTVisualizationTool.main()
