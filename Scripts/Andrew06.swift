import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation

import PenguinStructures

// Andrew06: Determine ability of densities to determine likelihood of unseen data
struct Andrew06: ParsableCommand {
  @Option(help: "Run for number of frames")
  var trackLength: Int = 80
  
  @Option(help: "Size of feature space")
  var featureSize: Int = 16

  @Option(help: "Pretrained weights")
  var weightsFile: String?

  func run() {
    let np = Python.import("numpy")
    let plt = Python.import("matplotlib.pyplot")
    let pickle = Python.import("pickle")

    let (imageHeight, imageWidth, imageChannels) =
      (40, 70, 1)

    let dataDir = URL(fileURLWithPath: "./OIST_Data")
    let data = OISTBeeVideo(directory: dataDir, length: 80)!
    let valData = OISTBeeVideo(directory: dataDir, afterIndex: 80, length: 20)!

    let (fg, bg, _) = getTrainingBatches(
        dataset: data, boundingBoxSize: (40, 70),
        fgBatchSize: 3000,
        bgBatchSize: 3000,
        fgRandomFrameCount: 20,
        bgRandomFrameCount: 20,
        useCache: false
    )

    let (valFg, valBg, _) = getTrainingBatches(
        dataset: valData, boundingBoxSize: (40, 70),
        fgBatchSize: 750,
        bgBatchSize: 750,
        fgRandomFrameCount: 20,
        bgRandomFrameCount: 20,
        useCache: false
    )


    let featureDimensions = [16, 64, 256]
    for (i, d) in featureDimensions.enumerated() {
        // Setting up RAE
        var rae = DenseRAE(
            imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels,
            hiddenDimension: [256, 256, 512][i], latentDimension: d
        )

        if let weightsFile = weightsFile {
            rae.load(weights: np.load(weightsFile, allow_pickle: true))
        } else {
            rae.load(weights: np.load("./oist_rae_weight_\(d).npy", allow_pickle: true))
        }

        let raeBatchNegative = rae.encode(bg)
        let raeBackgroundModel = MultivariateGaussian(from: raeBatchNegative, regularizer: 1e-3)
        let raeNBBackgroundModel = GaussianNB(from: raeBatchNegative, regularizer: 1e-3)

        let rp = RandomProjection(fromShape: TensorShape([imageHeight, imageWidth, imageChannels]), toFeatureSize: d)
        let rpBatchNegative = rp.encode(bg)
        let rpBackgroundModel = MultivariateGaussian(from: rpBatchNegative, regularizer: 1e-3)
        let rpNBBackgroundModel = GaussianNB(from: rpBatchNegative, regularizer: 1e-3)

        var statistics = FrameStatistics(Tensor<Double>(0.0))
        statistics.mean = Tensor(62.26806976644069)
        statistics.standardDeviation = Tensor(37.44683834503672)
        let ppca = PCAEncoder(from: data.makeBatch(statistics: statistics, appearanceModelSize: (imageHeight, imageWidth), batchSize: 3000), given: d)

        let ppcaBatchNegative = ppca.encode(bg)
        let ppcaBackgroundModel = MultivariateGaussian(from: ppcaBatchNegative, regularizer: 1e-3)
        let ppcaNBBackgroundModel = GaussianNB(from: ppcaBatchNegative, regularizer: 1e-3)

        var raeMGLikelihood = 0.0
        var raeNBLikelihood = 0.0
        var rpMGLikelihood = 0.0
        var rpNBLikelihood = 0.0
        var ppcaMGLikelihood = 0.0
        var ppcaNBLikelihood = 0.0

        let raeEncodedVal = rae.encode(valBg)
        let rpEncodedVal = rp.encode(valBg)
        let ppcaEncodedVal = ppca.encode(valBg)
        for imageNum in (0..<valBg.shape[0]) {
            raeMGLikelihood += exp(-raeBackgroundModel.negativeLogLikelihood(raeEncodedVal[imageNum]))
            raeNBLikelihood += exp(-raeNBBackgroundModel.negativeLogLikelihood(raeEncodedVal[imageNum]))
            rpMGLikelihood += exp(-rpBackgroundModel.negativeLogLikelihood(rpEncodedVal[imageNum]))
            rpNBLikelihood += exp(-rpNBBackgroundModel.negativeLogLikelihood(rpEncodedVal[imageNum]))
            ppcaMGLikelihood += exp(-ppcaBackgroundModel.negativeLogLikelihood(ppcaEncodedVal[imageNum]))
            ppcaNBLikelihood += exp(-ppcaNBBackgroundModel.negativeLogLikelihood(ppcaEncodedVal[imageNum]))
        }
        print("RAE \(d), MG \(raeMGLikelihood/Double(valBg.shape[0]))                  NB \(raeNBLikelihood/Double(valBg.shape[0]))")
        print("RP \(d), MG \(rpMGLikelihood/Double(valBg.shape[0]))                  NB \(rpNBLikelihood/Double(valBg.shape[0]))")
        print("PPCA \(d), MG \(ppcaMGLikelihood/Double(valBg.shape[0]))                  NB \(ppcaNBLikelihood/Double(valBg.shape[0]))")

        let (fig, axes) = plt.subplots(figsize: Python.tuple([6, 6])).tuple2
        axes.hist(rp.encode(bg).makeNumpyArray()[np.arange(bg.shape[0]), 0], color:"g", alpha:0.5, label:"train")
        axes.hist(rp.encode(valBg).makeNumpyArray()[np.arange(valBg.shape[0]), 0], color:"r", alpha:0.5, label:"val")
        //axes[1].hist(rp.encode(valBg).makeNumpyArray()[np.arange(valBg.shape[0]), 0])
        axes.legend()
        fig.savefig("./rp_\(d)_histograms", bbox_inches: "tight")

    }
  
}
}

/// Returns `t` as a Swift tuple.
fileprivate func unpack<A, B>(_ t: Tuple2<A, B>) -> (A, B) {
  return (t.head, t.tail.head)
}
/// Returns `t` as a Swift tuple.
fileprivate func unpack<A>(_ t: Tuple1<A>) -> (A) {
  return (t.head)
}