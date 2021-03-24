import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation

import PenguinStructures

/// Andrew03: t-SNE to determine how the encoded images separate in higher dimensions
struct Andrew03: ParsableCommand {
  typealias LikelihoodModel = TrackingLikelihoodModel<PretrainedDenseRAE, MultivariateGaussian, MultivariateGaussian>
  @Option(help: "Run for number of frames")
  var trackLength: Int = 80
  
  @Option(help: "Size of feature space")
  var featureSize: Int = 256

  @Option(help: "Pretrained weights")
  var weightsFile: String?
  
  
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

  func run() {
    let np = Python.import("numpy")
    let plt = Python.import("matplotlib.pyplot")
    let pickle = Python.import("pickle")
    let tsne = Python.import("sklearn.manifold")
    let sns = Python.import("seaborn")
    let kHiddenDimension = 512
    plt.rcParams.update(Python.dict([
    "text.usetex": true
    ]))
    plt.rcParams.update(Python.dict([
        "text.usetex": true
    ]))

    let (imageHeight, imageWidth, imageChannels) =
      (40, 70, 1)
    let trainingDatasetSize = 100
    let dataDir = URL(fileURLWithPath: "./OIST_Data")
    let data = OISTBeeVideo(directory: dataDir, afterIndex: 100, length: 80)!


    var rae = DenseRAE(
      imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels,
      hiddenDimension: kHiddenDimension, latentDimension: featureSize
    )

    if let weightsFile = weightsFile {
      rae.load(weights: np.load(weightsFile, allow_pickle: true))
    } else {
      rae.load(weights: np.load("./oist_rae_weight_\(featureSize).npy", allow_pickle: true))
    }

    let (fg, bg, _) = getTrainingBatches(
    dataset: data, boundingBoxSize: (40, 70),
    fgBatchSize: 3000,
    bgBatchSize: 3000,
    fgRandomFrameCount: 50,
    bgRandomFrameCount: 50,
    useCache: false
  )

    np.random.seed(seed: 42)
    let batchPositive = rae.encode(fg)
    let foregroundModel = MultivariateGaussian(from:batchPositive, regularizer: 1e-3)
    let batchNegative = rae.encode(bg)
    let backgroundModel = MultivariateGaussian(from: batchNegative, regularizer: 1e-3)
    let nbBackgroundModel = GaussianNB(from: batchNegative, regularizer: 1e-3)
    
    let fg_encoded_output = batchPositive.makeNumpyArray()
    let bg_encoded_output = batchNegative.makeNumpyArray()
    let tsne_model = tsne.TSNE(n_components: 2, verbose: 1, perplexity: 50, n_iter: 2000, random_state:42)
    let tsne_results = tsne_model.fit_transform(X: np.concatenate([fg_encoded_output, bg_encoded_output]))

    
    let (fig, axes) = plt.subplots(1, 1, figsize: Python.tuple([6, 6])).tuple2
    var posLabel = np.full(fg_encoded_output.shape[0], "Foreground Patches")
    var negLabel = np.full(bg_encoded_output.shape[0], "Background Patches")
    let labels = np.concatenate([posLabel, negLabel])
    sns.scatterplot(
        x:tsne_results[np.arange(tsne_results.shape[0]), 0], y:tsne_results[np.arange(tsne_results.shape[0]), 1], hue: labels, palette: sns.color_palette("hls", 2), alpha: 0.8, s:10, ax: axes
    )

    axes.set_ylim([-100, 100])
    axes.set_xlim([-100, 100])
    axes.set_xlabel("t-SNE 1")
    axes.set_ylabel("t-SNE 2")
    fig.savefig("Results/andrew01/tsne.pdf", bbox_inches: "tight")
    plt.close("all")

      }
  
}