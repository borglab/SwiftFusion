import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation

/// Fan14: Aggregate data
struct Fan14: ParsableCommand {

  @Option(help: "Run for number of frames")
  var trackLength: Int = 80

  @Option(help: "Size of feature space")
  var featureSize: Int = 20

  @Flag(help: "Training mode")
  var training: Bool = false

  func readExperimentData(name: String) -> SubsequenceMetrics {
    let decoder = JSONDecoder()
    let trackPath = "Results/fan13/\(name)_track.json"
    let gtPath = "Results/fan13/\(name)_gt.json"
    let decodedTrack = try! decoder.decode([Pose2].self, from: Data(contentsOf: URL(fileURLWithPath: trackPath))).map { OrientedBoundingBox(center: $0, rows: 40, cols: 70)}
    let decodedGt = try! decoder.decode([Pose2].self, from: Data(contentsOf: URL(fileURLWithPath: gtPath))).map { OrientedBoundingBox(center: $0, rows: 40, cols: 70)}
    return SubsequenceMetrics(groundTruth: decodedGt, prediction: decodedTrack)
  }

  func run() {
    var metrics: [String: [SubsequenceMetrics]] = [:]

    let frameIds = 0..<10

    metrics["RP"] = frameIds.map { trackId in
      let exprNameNoEM = "fan13_rp_mg_mg_noem_track\(trackId)_\(featureSize)"

      return readExperimentData(name: exprNameNoEM)
    }

    metrics["PCA"] = frameIds.map { trackId in
      let exprNameNoEM = "fan13_pca_mg_mg_noem_track\(trackId)_\(featureSize)"

      return readExperimentData(name: exprNameNoEM)
    }
    
    metrics["AE"] = frameIds.map { trackId in
      let exprNameNoEM = "fan13_ae_mg_mg_noem_track\(trackId)_\(featureSize)"

      return readExperimentData(name: exprNameNoEM)
    }

    metrics["PCA+EM"] = frameIds.map { trackId in
      let exprNameNoEM = "fan13_pca_mg_mg_em_track\(trackId)_\(featureSize)"

      return readExperimentData(name: exprNameNoEM)
    }
    
    metrics["AE+EM"] = frameIds.map { trackId in
      let exprNameWithEM = "fan13_ae_mg_mg_track\(trackId)_\(featureSize)"

      return readExperimentData(name: exprNameWithEM)
    }

    var ssm: [String: ExpectedAverageOverlap] = [:]

    metrics.forEach { ssm[$0] = ExpectedAverageOverlap($1) }
    
    if !FileManager.default.fileExists(atPath: "Results/fan14") {
      do {
        try FileManager.default.createDirectory(atPath: "Results/fan14", withIntermediateDirectories: true, attributes: nil)
      } catch {
        print(error.localizedDescription);
      }
    }

    // Now create trajectory and metrics plot 
    let plt = Python.import("matplotlib.pyplot")
    let (fig, ax) = plt.subplots(1, 1, figsize: Python.tuple([8,6])).tuple2
    
    ssm.forEach {
      ax.plot($1.curve, label: $0)
    }
    
    // ax.plot(Array(zip(ssmWithEM.curve, ssmNoEM.curve).map { $0 - $1 }), label: "Diff")
    ax.set_title("Expected Average Overlap")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Overlap")
    ax.legend()

    fig.savefig("Results/fan14/overlap_comp.pdf", bbox_inches: "tight")
    fig.savefig("Results/fan14/overlap_comp.png", bbox_inches: "tight")
  }
}
