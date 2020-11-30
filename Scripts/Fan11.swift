import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import PythonKit
import Foundation
import TensorFlow

/// Fan11: Error Landscape
struct Fan11: ParsableCommand {
  @Option(help: "Size of feature space")
  var featureSize: Int = 100

  @Option(help: "Which frame to show")
  var frameId: Int = 0

  @Option(help: "Which track to show")
  var trackId: Int = 0

  typealias LikelihoodModel = TrackingLikelihoodModel<DenseRAE, MultivariateGaussian, MultivariateGaussian>

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

  // Visualize error landscape of PCA
  // Make sure you have a folder `Results/fan11` before running
  func run() {
    let dataDir = URL(fileURLWithPath: "./OIST_Data")

    let np = Python.import("numpy")

      // train foreground and background model and create tracker
    // let trainingData = OISTBeeVideo(directory: dataDir, length: 100)!
    // let testData = OISTBeeVideo(directory: dataDir, afterIndex: 100, length: forFrames)!
    let (imageHeight, imageWidth, imageChannels) = (40, 70, 1)
    // let encoder = RandomProjection(fromShape: TensorShape([imageHeight, imageWidth, imageChannels]), toFeatureSize: featureSize)
  
    // let encoder = PCAEncoder(
    //   withBasis: Tensor<Double>(numpy: np.load("./pca_U_\(featureSize).npy"))!,
    //   andMean: Tensor<Double>(numpy: np.load("./pca_mu_\(featureSize).npy"))!
    // )
    var encoder = DenseRAE(
      imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels,
      hiddenDimension: 100, latentDimension: featureSize
    )

    encoder.load(weights: np.load("./oist_rae_weight_\(featureSize).npy", allow_pickle: true))

    // let (fg, bg, statistics) = getTrainingBatches(
    //   dataset: trainingData, boundingBoxSize: (40, 70),
    //   fgBatchSize: 3000,
    //   bgBatchSize: 3000,
    //   fgRandomFrameCount: 100,
    //   bgRandomFrameCount: 100,
    //   useCache: true
    // )

    // let batchPositive = encoder.encode(fg)
    // let foregroundModel = MultivariateGaussian(from: batchPositive, regularizer: 1e-3)

    // let batchNegative = encoder.encode(bg)
    // let backgroundModel = MultivariateGaussian(from: batchNegative, regularizer: 1e-3)

    let generator = ARC4RandomNumberGenerator(seed: 42)
    var em = MonteCarloEM<LikelihoodModel>(sourceOfEntropy: generator)
    
    let trainingDataset = OISTBeeVideo(directory: dataDir, length: 100)!
    
    let trainingData = getTrainingDataEM(from: trainingDataset, numberForeground: 300, numberBackground: 300)
    
    // precondition(trainingData.count == 800, "Wrong trainingData dims \(trainingData.count)")

    var statistics = FrameStatistics(Tensor<Double>(0.0))
    statistics.mean = Tensor(62.26806976644069)
    statistics.standardDeviation = Tensor(37.44683834503672)

    print("Running EM...")

    let trackingModel = em.run(
      with: trainingData,
      modelInitializer: { _,_ in
        TrackingLikelihoodModel(
          with: encoder,
          from: Tensor(
            stacking: trainingData.filter { $0.type == .fg }.map { statistics.normalized($0.frame!.patch(at: $0.obb)) }
          ), and: Tensor(
            stacking: trainingData.filter { $0.type == .bg }.map { statistics.normalized($0.frame!.patch(at: $0.obb)) }
          )
        )
      },
      modelFitter: { _ in
        TrackingLikelihoodModel(
          with: encoder,
          from: Tensor(
            stacking: trainingData.filter { $0.type == .fg }.map { statistics.normalized($0.frame!.patch(at: $0.obb)) }
          ), and: Tensor(
            stacking: trainingData.filter { $0.type == .bg }.map { statistics.normalized($0.frame!.patch(at: $0.obb)) }
          )
        )
      },
      iterationCount: 5,
      sampleCount: 10,
      hook: { i, data, _ in
        print("EM Iteration \(i)")
        let percentage_fg = data.filter { i in
          if case LikelihoodModel.Hidden.fg = data[0].0 {
            return true
          } else {
            return false
          }
        }.count
        print("Label Positive \(percentage_fg)")
      }
    )

    let deltaXRange = Array(-60..<60).map { Double($0) }
    let deltaYRange = Array(-40..<40).map { Double($0) }

    let datasetToShow = OISTBeeVideo(directory: dataDir, afterIndex: frameId - 1, length: 2)!
    let frame = datasetToShow.frames[1]
    let pose = datasetToShow.tracks[trackId].boxes[0].center
    let (fig, _) = plotErrorPlaneTranslation(
      frame: frame,
      at: pose,
      deltaXs: deltaXRange,
      deltaYs: deltaYRange,
      statistics: statistics,
      encoder: encoder,
      foregroundModel: trackingModel.foregroundModel,
      backgroundModel: trackingModel.backgroundModel
    )
    fig.savefig("Results/fan11/fan11_em_ae_mg_mg_\(trackId)_\(frameId)_\(featureSize).pdf", bbox_inches: "tight")
    fig.savefig("Results/fan11/fan11_em_ae_mg_mg_\(trackId)_\(frameId)_\(featureSize).png", bbox_inches: "tight")
  }
}
