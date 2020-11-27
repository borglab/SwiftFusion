import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation

typealias LikelihoodModel = TrackingLikelihoodModel<DenseRAE, MultivariateGaussian, GaussianNB>

/// Fan10: RP Tracker, using the new tracking model
struct Fan10: ParsableCommand {
  @Option(help: "Run on track number x")
  var trackId: Int = 0

  @Option(help: "Run for number of frames")
  var trackLength: Int = 80

  @Option(help: "Size of feature space")
  var featureSize: Int = 30

  @Flag(help: "Training mode")
  var training: Bool = false

  func getTrainingDataEM(
    from dataset: OISTBeeVideo,
    numberForeground: Int = 3000,
    numberBackground: Int = 3000
  ) -> [LikelihoodModel.Datum] {
    let bgBoxes = dataset.makeBackgroundBoundingBoxes(patchSize: (40, 70), batchSize: numberBackground).map {
      (frame: $0.frame, type: LikelihoodModel.PatchType.bg, obb: $0.obb)
    }
    let fgBoxes = dataset.makeForegroundBoundingBoxes(patchSize: (40, 70), batchSize: numberForeground).map {
      (frame: $0.frame, type: LikelihoodModel.PatchType.fg, obb: $0.obb)
    }
    
    return fgBoxes + bgBoxes
  }
  
  // Just runs an RP tracker and saves image to file
  // Make sure you have a folder `Results/fan10` before running
  func run() {
    let kHiddenDimension = 100
    let np = Python.import("numpy")
    let dataDir = URL(fileURLWithPath: "./OIST_Data")

    let (imageHeight, imageWidth, imageChannels) =
      (40, 70, 1)
    
    var rae = DenseRAE(
      imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels,
      hiddenDimension: kHiddenDimension, latentDimension: featureSize
    )

    rae.load(weights: np.load("./oist_rae_weight_\(featureSize).npy", allow_pickle: true))

    var generator = ARC4RandomNumberGenerator(seed: 42)
    var em = MonteCarloEM<LikelihoodModel>(sourceOfEntropy: generator)
    
    let trainingDataset = OISTBeeVideo(directory: dataDir, length: 100)!
    
    let trainingData = getTrainingDataEM(from: trainingDataset)
    
    let trackingModel = em.run(with: trainingData, iterationCount: 3)
    
    let (fig, track, gt) = runProbabilisticTracker(
      directory: dataDir,
      encoder: rae,
      onTrack: trackId, forFrames: trackLength, withSampling: true,
      withFeatureSize: featureSize,
      savePatchesIn: "Results/fan10"
    )

    /// Actual track v.s. ground truth track
    fig.savefig("Results/fan10/fan10_track\(trackId)_\(featureSize).pdf", bbox_inches: "tight")
    fig.savefig("Results/fan10/fan10_track\(trackId)_\(featureSize).png", bbox_inches: "tight")

    let json = JSONEncoder()
    json.outputFormatting = .prettyPrinted

    let track_data = try! json.encode(track)
    try! track_data.write(to: URL(fileURLWithPath: "Results/fan10/fan10_track_\(trackId)_\(featureSize).json"))

    let gt_data = try! json.encode(gt)
    try! gt_data.write(to: URL(fileURLWithPath: "Results/fan10/fan10_gt_\(trackId)_\(featureSize).json"))
  }
}
