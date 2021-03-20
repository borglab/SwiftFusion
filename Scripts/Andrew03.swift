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
    }//.filter{$0.obb.center.rot.theta != -1.5707963267948966}
    
    return fgBoxes + bgBoxes
  }


  // Runs RAE tracker on n number of sequences and outputs relevant images and statistics
  // Make sure you have a folder `Results/andrew01` before running
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

    // EM Code
    // let generator = ARC4RandomNumberGenerator(seed: 42)
    
    // var em = MonteCarloEM<LikelihoodModel>(sourceOfEntropy: generator)
    // var statistics = FrameStatistics(Tensor<Double>(0.0))
    // statistics.mean = Tensor(62.26806976644069)
    // statistics.standardDeviation = Tensor(37.44683834503672)
    // let likelihoodModel = em.run(
    //   with: getTrainingDataEM(from: data),
    //   iterationCount: 3,
    //   hook: { i, _, _ in
    //     print("EM run iteration \(i)")
    //   },
    //   given: LikelihoodModel.HyperParameters(
    //     encoder: PretrainedDenseRAE.HyperParameters(hiddenDimension: kHiddenDimension, latentDimension: featureSize, weightFile: "./oist_rae_weight_\(featureSize).npy"), frameStatistics: statistics
    //   )
    // )


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

    let normalizedPositive = matmul(batchPositive - foregroundModel.mean, foregroundModel.information) - matmul(batchPositive - backgroundModel.mean, backgroundModel.information)
    let nbNormalizedPositive = matmul(batchPositive - foregroundModel.mean, foregroundModel.information) - (batchNegative - nbBackgroundModel.mu) / nbBackgroundModel.sigmas.squared()
    //let emNormalizedPositive = matmul(likelihoodModel.encoder.encode(fg) - likelihoodModel.foregroundModel.mean, likelihoodModel.foregroundModel.information) - matmul(likelihoodModel.encoder.encode(fg) - likelihoodModel.backgroundModel.mean, likelihoodModel.backgroundModel.information) 

    let normalizedNegative = matmul(batchNegative - foregroundModel.mean, foregroundModel.information) - matmul(batchNegative - backgroundModel.mean, backgroundModel.information)
    let nbNormalizedNegative = matmul(batchNegative - foregroundModel.mean, foregroundModel.information) - (batchNegative - nbBackgroundModel.mu) / nbBackgroundModel.sigmas.squared()
    //let emNormalizedNegative = matmul(likelihoodModel.encoder.encode(bg) - likelihoodModel.foregroundModel.mean, likelihoodModel.foregroundModel.information) - matmul(likelihoodModel.encoder.encode(bg) - likelihoodModel.backgroundModel.mean, likelihoodModel.backgroundModel.information) 

    let fg_encoded_output = batchPositive.makeNumpyArray()
    let bg_encoded_output = batchNegative.makeNumpyArray()
    let fg_nb_encoded_output = nbNormalizedPositive.makeNumpyArray()
    let bg_nb_encoded_output = nbNormalizedNegative.makeNumpyArray()
    //let fg_em_encoded_output = emNormalizedPositive.makeNumpyArray()
    //let bg_em_encoded_output = emNormalizedNegative.makeNumpyArray()
    let tsne_model = tsne.TSNE(n_components: 2, verbose: 1, perplexity: 50, n_iter: 2000, random_state:42)
    let tsne_results = tsne_model.fit_transform(X: np.concatenate([fg_encoded_output, bg_encoded_output]))
    //let nb_tsne_results = tsne_model.fit_transform(X: np.concatenate([fg_nb_encoded_output, bg_nb_encoded_output]))

    // var indices: Array<Int> = []
    // for i in (0..<Int(tsne_results.shape[0])!) {
    //   if abs(tsne_results[i, 0]) + abs(tsne_results[i, 1]) < 2.0 {
    //     indices.append(i)
    //     let (fig, ax) = plt.subplots(figsize: Python.tuple([4, 4])).tuple2
    //     if i >= 3000 {
    //       ax.imshow(bg[i - 3000].makeNumpyArray(), cmap: "gray")
    //       fig.savefig("Results/andrew01/t-sne-images/\(i).png", bbox_inches: "tight")
    //     }
    //     else {
    //       ax.imshow(fg[i].makeNumpyArray(), cmap: "gray")
    //       fig.savefig("Results/andrew01/t-sne-images/\(i).png", bbox_inches: "tight")
    //     }
        
    //   }
    // }
    // print(indices)
    // //let em_tsne_results = tsne_model.fit_transform(X: np.concatenate([fg_em_encoded_output, bg_em_encoded_output]))

    // // plt.tick_params(
    // // axis:"x",         
    // // which:"both",      
    // // bottom: false,      
    // // top: false,        
    // // labelbottom: false) 
    // // plt.tick_params(
    // // axis:"y",         
    // // which:"both",      
    // // bottom: false,      
    // // top: false,        
    // // labelbottom: false) 
    
    let (fig, axes) = plt.subplots(1, 1, figsize: Python.tuple([6, 6])).tuple2
    var posLabel = np.full(fg_encoded_output.shape[0], "Foreground Patches")
    //posLabel[np.arange(fg_encoded_output.shape[0])] = "Foreground Images"
    var negLabel = np.full(bg_encoded_output.shape[0], "Background Patches")
    //negLabel[np.arange(bg_encoded_output.shape[0])] = "Background Images"
    let labels = np.concatenate([posLabel, negLabel])
    sns.scatterplot(
        x:tsne_results[np.arange(tsne_results.shape[0]), 0], y:tsne_results[np.arange(tsne_results.shape[0]), 1], hue: labels, palette: sns.color_palette("hls", 2), alpha: 0.8, s:10, ax: axes
    )
    // sns.scatterplot(
    //     x:nb_tsne_results[np.arange(nb_tsne_results.shape[0]), 0], y:nb_tsne_results[np.arange(nb_tsne_results.shape[0]), 1], hue: labels, palette: sns.color_palette("hls", 2), alpha: 0.8, s:10, ax: axes[1]
    // )
    //sns.scatterplot(
    //    x:em_tsne_results[np.arange(em_tsne_results.shape[0]), 0], y:em_tsne_results[np.arange(em_tsne_results.shape[0]), 1], hue: labels, palette: sns.color_palette("hls", 2), alpha: 0.5, s:10, ax: axes[2]
    //)
    //plt.setp(ax.get_legend().get_texts())
    //fig.title.set_text("T-SNE Visualization of the ")
    //fig.set(xlabel: "T-SNE 1", ylabel: "T-SNE 2")
    // np.save("Results/andrew01/t-sne-images/bg.npy", bg.makeNumpyArray())
    // np.save("Results/andrew01/t-sne-images/fg.npy", fg.makeNumpyArray())
    // np.save( "Results/andrew01/t-sne-images/bg_encoded_output.npy", bg_encoded_output)
    // np.save("Results/andrew01/t-sne-images/fg_encoded_output.npy", fg_encoded_output)
    // np.save("Results/andrew01/t-sne-images/fg_nb_encoded_output.npy", fg_nb_encoded_output)
    // np.save("Results/andrew01/t-sne-images/bg_nb_encoded_output.npy",bg_nb_encoded_output)
    // np.save("Results/andrew01/t-sne-images/tsne_results.npy", tsne_results)
    // np.save("Results/andrew01/t-sne-images/nb_tsne_results.npy", nb_tsne_results)
    axes.set_ylim([-100, 100])
    axes.set_xlim([-100, 100])
    //axes[1].set_ylim([-100, 100])
    //axes[1].set_xlim([-100, 100])
    axes.set_xlabel("t-SNE 1")
    axes.set_ylabel("t-SNE 2")
    //axes[1].set_xlabel("t-SNE 1")
    //axes[1].set_ylabel("t-SNE 2")
    fig.savefig("Results/andrew01/tsne.pdf", bbox_inches: "tight")
    plt.close("all")

      }
  
}