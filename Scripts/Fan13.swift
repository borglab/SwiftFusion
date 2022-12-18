import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
// import TensorFlow
import PythonKit
import Foundation

/// Fan13: RP Tracker, using the new tracking model
struct Fan13: ParsableCommand {

//   @Option(help: "Run on track number x")
//   var trackId: Int = 3

  @Option(help: "Run for number of frames")
  var trackLength: Int = 80

  @Option(help: "Size of feature space")
  var featureSize: Int = 20

  @Flag(help: "Training mode")
  var training: Bool = false

  func getTrainingDataEM(
    from dataset: OISTBeeVideo,
    numberForeground: Int = 300,
    numberBackground: Int = 300
  ) -> [LikelihoodModel.Datum] {
    let bgBoxes = dataset.makeBackgroundBoundingBoxes(patchSize: (40, 70), batchSize: numberBackground).map {
      (frame: $0.frame, type: LikelihoodModel.PatchType.bg, obb: $0.obb)
    }
    let fgBoxes = dataset.makeForegroundBoundingBoxes(patchSize: (40, 70), batchSize: numberForeground).map {
      (frame: $0.frame, type: LikelihoodModel.PatchType.fg, obb: $0.obb)
    }
    
    return fgBoxes + bgBoxes
  }
    
  typealias LikelihoodModel = TrackingLikelihoodModel<PCAEncoder, MultivariateGaussian, MultivariateGaussian>

  // Just runs an RP tracker and saves image to file
  // Make sure you have a folder `Results/fan13` before running
  func run() {
    let kHiddenDimension = 100
    let dataDir = URL(fileURLWithPath: "./OIST_Data")

    let generator = ARC4RandomNumberGenerator(seed: 42)
    var wrapped = AnyRandomNumberGenerator(generator)
    var em = MonteCarloEM<LikelihoodModel>(sourceOfEntropy: generator)
    
    let trainingDataset = OISTBeeVideo(directory: dataDir, length: 30)!
    
    let trainingData = getTrainingDataEM(from: trainingDataset)
    
    let encoderType = "pca"

    // let trackingModel = em.run(
    //   with: trainingData,
    //   iterationCount: 3,
    //   hook: { i, _, _ in
    //     print("EM run iteration \(i)")
    //   },
    //   given: LikelihoodModel.HyperParameters(
    //     encoder: PretrainedDenseRAE.HyperParameters(hiddenDimension: kHiddenDimension, latentDimension: featureSize, weightFile: "./oist_rae_weight_\(featureSize).npy")
    //   )
    // )

    // let trackingModel = LikelihoodModel(from: trainingData, using: &wrapped, given: LikelihoodModel.HyperParameters(
    //   encoder: PretrainedDenseRAE.HyperParameters(hiddenDimension: kHiddenDimension, latentDimension: featureSize, weightFile: "./oist_rae_weight_\(featureSize).npy")
    // ))

    // let trackingModel = LikelihoodModel(from: trainingData, using: &wrapped, given: LikelihoodModel.HyperParameters(
    //   encoder: PCAEncoder.HyperParameters(featureSize)
    // ))

    let emType = "em"
    let trackingModel = em.run(
      with: trainingData,
      iterationCount: 3,
      hook: { i, _, _ in
        print("EM run iteration \(i)")
      },
      given: LikelihoodModel.HyperParameters(
        encoder: PCAEncoder.HyperParameters(featureSize)
      )
    )

    for trackId in 0..<19 {
      let exprName = "fan13_\(encoderType)_mg_mg_\(emType)_track\(trackId)_\(featureSize)"
      let imagesPath = "Results/fan13/\(exprName)"
      if !FileManager.default.fileExists(atPath: imagesPath) {
        do {
          try FileManager.default.createDirectory(atPath: imagesPath, withIntermediateDirectories: true, attributes: nil)
        } catch {
          print(error.localizedDescription);
        }
      }
      
      let (fig, track, gt) = runProbabilisticTracker(
        directory: dataDir,
        likelihoodModel: trackingModel,
        onTrack: trackId, forFrames: trackLength, withSampling: true,
        withFeatureSize: featureSize,
        savePatchesIn: "Results/fan13/\(exprName)"
      )

      /// Actual track v.s. ground truth track
      fig.savefig("Results/fan13/\(exprName).pdf", bbox_inches: "tight")
      fig.savefig("Results/fan13/\(exprName).png", bbox_inches: "tight")

      let json = JSONEncoder()
      json.outputFormatting = .prettyPrinted

      let track_data = try! json.encode(track)
      try! track_data.write(to: URL(fileURLWithPath: "Results/fan13/\(exprName)_track.json"))

      let gt_data = try! json.encode(gt)
      try! gt_data.write(to: URL(fileURLWithPath: "Results/fan13/\(exprName)_gt.json"))
    }
  }
}
