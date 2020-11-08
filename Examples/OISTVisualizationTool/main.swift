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
  @Option(help: "Location of dataset folder which should contain `frames` and `frames_txt`")
  var datasetLocation: String = "./OIST_Data"

  @Option(help: "Which frame to show")
  var frameId: Int = 0

  func run() {
    let dataURL = URL(fileURLWithPath: datasetLocation)
    print("Viewing \(dataURL) at frame \(frameId)")
    let dataset = OISTBeeVideo(directory: dataURL, deferLoadingFrames: true)!

    let frameRawId = dataset.frameIds[frameId]

    let image = dataset.loadFrame(frameRawId)!

    plotImagePlotly(image, boxes: dataset.labels[frameId].enumerated().map {
      (String($0), $1.location)
    }, margin: 10.0, scale: 0.5).show()
  }
}

OISTVisualizationTool.main()
