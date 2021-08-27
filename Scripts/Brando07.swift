import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation

import PenguinStructures

/// Brando07: RAE + Prob density histograms
struct Brando07: ParsableCommand {
  @Option(help: "Run for number of frames")
  var trackLength: Int = 80
  
  @Option(help: "Size of feature space")
  var featureSize: Int = 256
  // used to be 256

  @Option(help: "Pretrained weights")
  var weightsFile: String?

  // Runs RAE tracker on n number of sequences and outputs relevant images and statistics
  func run() {
    let np = Python.import("numpy")
    let kHiddenDimension = 512
    // used to be 512

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
    print("s")

    let trainingDatasetSize = 100

    let dataDir = URL(fileURLWithPath: "./OIST_Data")
    let numberOfTrainingSamples = 3000
    // let fgRandomFrameCount = 10
    // let bgRandomFrameCount = 10
    // let boundingBoxSize = (40, 70)

    let dataset = OISTBeeVideo(directory: dataDir, length: 100)! // calling this twice caused the Killed to happen
    let batchSize = 3000
    // print("tests here1")
    let fgBoxes = dataset.makeForegroundBoundingBoxes(patchSize: (40, 70), batchSize: batchSize)
    print("here 1.5")
    let bgBoxes = dataset.makeBackgroundBoundingBoxes(patchSize: (40, 70), batchSize: batchSize)
    print("tests here2")
    let fgpatches = Tensor<Double>(stacking: fgBoxes.map { $0.frame!.patch(at: $0.obb)})
    let bgpatches = Tensor<Double>(stacking: bgBoxes.map { $0.frame!.patch(at: $0.obb)})
    print("patches complete")

    // let (fg, bg, _) = getTrainingBatches(
    //     dataset: dataset, boundingBoxSize: boundingBoxSize,
    //     fgBatchSize: numberOfTrainingSamples,
    //     bgBatchSize: numberOfTrainingSamples,
    //     fgRandomFrameCount: fgRandomFrameCount,
    //     bgRandomFrameCount: bgRandomFrameCount,
    //     useCache: true
    // )

    let batchPositive = rae.encode(fgpatches)
    print("shape batch positive", batchPositive.shape)
    // let foregroundModel = GaussianNB(from:batchPositive, regularizer: 1e-3)
    let foregroundModel = MultivariateGaussian(from:batchPositive, regularizer: 1e-3)
    let batchNegative = rae.encode(bgpatches)
    // let backgroundModel = GaussianNB(from: batchNegative, regularizer: 1e-3)
    let backgroundModel = MultivariateGaussian(from: batchNegative, regularizer: 1e-3)

    var outfg0 = [Double]()
    var outfg1 = [Double]()
    var outbg0 = [Double]()
    var outbg1 = [Double]()
    print(batchPositive[0,0...].shape)
    print(backgroundModel.probability(batchPositive[0,0...]))
    print(foregroundModel.probability(batchPositive[0,0...]))

    for i in 0...numberOfTrainingSamples-1 {
        outfg0.append(backgroundModel.probability(batchPositive[i,0...]))
        // print("probability", backgroundModel.probability(batchPositive[i,0...]))
        outfg1.append(foregroundModel.probability(batchPositive[i,0...]))
        outbg0.append(backgroundModel.probability(batchNegative[i,0...]))
        outbg1.append(foregroundModel.probability(batchNegative[i,0...]))
    }
    // print(outfg0)
    // print(outfg1)

    // let batchSize = numberOfTrainingSamples
    var plt = Python.import("matplotlib.pyplot")


    var fgsum0 = 0.0
    var fgsum1 = 0.0
    var bgsum0 = 0.0
    var bgsum1 = 0.0
    var fg0_arr = [Double]()
    var fg1_arr = [Double]()
    var bg0_arr = [Double]()
    var bg1_arr = [Double]()
    for i in 0...batchSize-1 {
        fgsum0 += (outfg0[i])
        fgsum1 += (outfg1[i])
        bgsum0 += (outbg0[i])
        bgsum1 += (outbg1[i])
        fg0_arr.append((outfg0[i]))
        fg1_arr.append((outfg1[i]))
        bg0_arr.append((outbg0[i]))
        bg1_arr.append((outbg1[i]))
    }
    print("featSize", featureSize, "kHiddendimension", kHiddenDimension, "val", fgsum1 + bgsum0 - fgsum0 - bgsum1)




    print("feature size", featureSize)
    print("fgsum1", fgsum1, "fgsum0", fgsum0)
    print("bgsum1", bgsum1, "bgsum0", bgsum0)

    var (figs, axs) = plt.subplots(2,2).tuple2
    print("asda")
    // plt.GridSpec(2, 2, wspace: 0.1, hspace: 0.8)

    plt.subplots_adjust(left:0.1,
            bottom:0.1, 
            right:0.9, 
            top:0.9, 
            wspace:0.4, 
            hspace:0.4)


    // var (fig, ax1) = plt.subplots().tuple2
    var ax1 = axs[1,0]
    ax1.hist(fg0_arr, range: Python.tuple([-1,1]), bins: 50)
    var mean = fgsum0/Double(batchSize)
    var sd = 0.0
    for elem in fg0_arr {
        sd += abs(elem - mean)/Double(batchSize)
    }
    ax1.set_title("Foreground. Output response for background. \n Mean = \(String(format: "%.2f", mean)) and SD = \(sd).", fontsize:8)

    // (fig, ax1) = plt.subplots().tuple2
    ax1 = axs[0,0]
    ax1.hist(fg1_arr, range: Python.tuple([-1,1]), bins: 50)
    mean = fgsum1/Double(batchSize)
    sd = 0.0
    for elem in fg1_arr {
        sd += abs(elem - mean)/Double(batchSize)
    }
    ax1.set_title("Foreground. Output response for foreground. \n Mean = \(String(format: "%.2f", mean)) and SD = \(sd).", fontsize:8)

    ax1 = axs[1,1]
    // (fig, ax1) = plt.subplots().tuple2
    ax1.hist(bg0_arr, range: Python.tuple([-1,1]), bins: 50)
    mean = bgsum0/Double(batchSize)
    sd = 0.0
    for elem in bg0_arr {
        sd += abs(elem - mean)/Double(batchSize)
    }
    ax1.set_title("Background. Output response for background. \n Mean = \(String(format: "%.2f", mean)) and SD = \(sd).", fontsize:8)

    ax1 = axs[0,1]

    // (fig, ax1) = plt.subplots().tuple2
    ax1.hist(bg1_arr, range: Python.tuple([-1,1]), bins: 50)
    mean = bgsum1/Double(batchSize)
    sd = 0.0
    for elem in bg1_arr {
        sd += abs(elem - mean)/Double(batchSize)
    }
    ax1.set_title("Background. Output response for foreground. \n Mean = \(String(format: "%.2f", mean)) and SD = \(sd).", fontsize:8)

    figs.savefig("hist_rae_\(kHiddenDimension)_\(featureSize).png")
    plt.close(figs)



  }





}


