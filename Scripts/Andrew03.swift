import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation

import PenguinStructures

/// Andrew01: RAE Tracker
struct Andrew03: ParsableCommand {
  @Option(help: "Run for number of frames")
  var trackLength: Int = 80
  
  @Option(help: "Size of feature space")
  var featureSize: Int = 256

  @Option(help: "Pretrained weights")
  var weightsFile: String?
    // typealias CurrentModel = ProbablisticTracker<PretrainedDenseRAE, MultivariateGaussian, MultivariateGaussian>
    // func getTrainingDataEM(
    // from dataset: OISTBeeVideo,
    // numberForeground: Int = 50,
    // numberBackground: Int = 50) -> [CurrentModel.Datum] {
    //     let bgBoxes = dataset.makeBackgroundBoundingBoxes(patchSize: (40, 70), batchSize: numberBackground).map {
    //     (frame: $0.frame, type: CurrentModel.PatchType.bg, obb: $0.obb)
    //     }
    //     let fgBoxes = dataset.makeForegroundBoundingBoxes(patchSize: (40, 70), batchSize: numberForeground).map {
    //     (frame: $0.frame, type: CurrentModel.PatchType.fg, obb: $0.obb)
    //     }
        
    //     return fgBoxes + bgBoxes
    // }
  // Runs RAE tracker on n number of sequences and outputs relevant images and statistics
  // Make sure you have a folder `Results/andrew01` before running
  func run() {
    let np = Python.import("numpy")
    let plt = Python.import("matplotlib.pyplot")
    let pickle = Python.import("pickle")
    let tsne = Python.import("sklearn.manifold")
    let sns = Python.import("seaborn")
    let kHiddenDimension = 512

    let (imageHeight, imageWidth, imageChannels) =
      (40, 70, 1)


    var rae = DenseRAE(
      imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels,
      hiddenDimension: kHiddenDimension, latentDimension: featureSize
    )

    if let weightsFile = weightsFile {
      rae.load(weights: np.load(weightsFile, allow_pickle: true))
    } else {
      rae.load(weights: np.load("./oist_rae_weight_\(featureSize).npy", allow_pickle: true))
    }
    
    let trainingDatasetSize = 100
    let dataDir = URL(fileURLWithPath: "./OIST_Data")
    let data = OISTBeeVideo(directory: dataDir, length: trainingDatasetSize)!

    let (fg, bg, _) = getTrainingBatches(
    dataset: data, boundingBoxSize: (40, 70),
    fgBatchSize: 3000,
    bgBatchSize: 3000,
    fgRandomFrameCount: 50,
    bgRandomFrameCount: 50,
    useCache: false
  )
    let fg_encoded_output = rae.encode(fg).makeNumpyArray()
    let bg_encoded_output = rae.encode(bg).makeNumpyArray()
    let tsne_model = tsne.TSNE(n_components: 2, verbose: 1, perplexity: 5, early_exaggeration: 5, n_iter: 2000)
    let tsne_results = tsne_model.fit_transform(X: np.concatenate([fg_encoded_output, bg_encoded_output]))

    var posLabel = np.full(fg_encoded_output.shape[0], "Foreground Patches")
    //posLabel[np.arange(fg_encoded_output.shape[0])] = "Foreground Images"
    var negLabel = np.full(bg_encoded_output.shape[0], "Background Patches")
    //negLabel[np.arange(bg_encoded_output.shape[0])] = "Background Images"
    let labels = np.concatenate([posLabel, negLabel])
    let fig = plt.figure(figsize: Python.tuple([7, 6]))
    let ax = sns.scatterplot(
        x:tsne_results[np.arange(tsne_results.shape[0]), 0], y:tsne_results[np.arange(tsne_results.shape[0]), 1], hue: labels, palette: sns.color_palette("hls", 2), alpha: 0.75, s:10
    )
    //plt.setp(ax.get_legend().get_texts())
    //fig.title.set_text("T-SNE Visualization of the ")
    //fig.set(xlabel: "T-SNE 1", ylabel: "T-SNE 2")
    plt.xlabel("T-SNE 1")
    plt.ylabel("T-SNE 2")
    fig.savefig("Results/andrew01/tsne.png", bbox_inches: "tight")
    plt.close("all")

      }
  
}