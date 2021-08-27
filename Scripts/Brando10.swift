


import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation

import PenguinStructures

/// Brando10: Plot the samplings in progress.
struct Brando10: ParsableCommand {
  @Option(help: "Run for number of frames")
  var trackLength: Int = 80

  // Runs NNClassifier tracker on n number of sequences and outputs relevant images and statistics
  func run() {
    let np = Python.import("numpy")
    let featureSizes = [512]
    let kHiddenDimensions = [512]
    let iterations = [1]
    let trainingDatasetSize = 100

    let dataDir = URL(fileURLWithPath: "./OIST_Data")
    let data = OISTBeeVideo(directory: dataDir, length: trainingDatasetSize)!
    let testData = OISTBeeVideo(directory: dataDir, afterIndex: trainingDatasetSize, length: trackLength)!

    let trackerEvaluation = TrackerEvaluationDataset(testData)

    for i in 0...78 {
        let folderName = "./sampling_512_512_2000samples"
        let posex_np = np.load(folderName + "/sampling_frame_\(i)_posex.npy")
        let posey_np = np.load(folderName + "/sampling_frame_\(i)_posey.npy")
        let posetheta_np = np.load(folderName + "/sampling_frame_\(i)_posetheta.npy")
        let error_np = np.load(folderName + "/sampling_frame_\(i)_error.npy")
        let t = np.arange(0, 2000, 1)
        

        let plt = Python.import("matplotlib.pyplot")
        var (figs, axs) = plt.subplots(1,1, figsize: Python.tuple([10, 4])).tuple2


        // axs[0].plot(t,posex_np, linewidth: 1)
        // axs[0].set_title("x and y coordinates")
        // axs[0].plot(t,posey_np, linewidth: 1)
        // axs[1].set_title("theta")
        // axs[1].plot(t,posetheta_np, linewidth: 1)
        axs[0].plot(t,error_np, linewidth: 1)
        axs[0].set_title("error")
        plt.subplots_adjust(left:0.1,
            bottom:0.1, 
            right:0.9, 
            top:0.9, 
            wspace:0.4, 
            hspace:0.4)
        // axs[2].setylim(-200,50)

        figs.savefig(folderName + "/sampling_figure_\(i).png")
        plt.close("all")

    }
    


    // for featureSize in featureSizes {
    // for kHiddenDimension in kHiddenDimensions {
    // for j in iterations {
        




    // }
    // }
    // }


    
  }
}