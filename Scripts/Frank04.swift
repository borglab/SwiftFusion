import ArgumentParser

import SwiftFusion
import BeeTracking
import BeeDataset
import PythonKit
import Foundation

/// Frank01: Random Projections Baseline Tracker
struct Frank04: ParsableCommand {

  func writeOneFile(filename: URL, labels: [OISTBeeLabel]) {
    var lines = ""
    for label in labels {
      let converted_label = OISTBeeLabel(
        frameIndex: label.frameIndex,
        label: label.label,
        rawLocation: (label.rawLocation.0 / 2.0, label.rawLocation.1 / 2.0, label.rawLocation.2),
        offset: (label.offset.0 / 2, label.offset.1 / 2),
        scale: 2.0
      )
      lines = lines.appending("\(converted_label.toString())\n")
    }
    
    do {
      try lines.write(to: filename, atomically: true, encoding: .utf8)
    } catch {
      print("error creating file")
    }
  }
  
  func run() {
    let dataset = OISTBeeVideo(deferLoadingFrames: true)!
    
    print(dataset.labels.count)
    for (index, labels) in dataset.labels.enumerated() {
      let frameId = dataset.frameIds[index]
      let filename = formOISTFilename(dataset.fps, frameId)
      print("Write \(filename)")
      writeOneFile(filename: URL(fileURLWithPath: "./converted/" + filename), labels: labels)
    }
    
  }
}

/// For output integers with padding
fileprivate extension String.StringInterpolation {
    mutating func appendInterpolation(_ val: Int, padding: UInt) {
        appendLiteral(String(format: "%0\(padding)d", arguments: [val]))
    }
}