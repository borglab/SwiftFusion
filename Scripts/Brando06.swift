import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation


import PenguinStructures

// This script produces HISTOGRAMS for the output of NN Classifiers
struct Brando06: ParsableCommand {    

    func run() {
        // let featSizes = [8,16,64,128,256]
        let dataDir = URL(fileURLWithPath: "./OIST_Data")
        let testData = OISTBeeVideo(directory: dataDir, afterIndex: 100, length: 80)!
        let batchSize = 3000
        // print("tests here1")
        let fgBoxes = testData.makeForegroundBoundingBoxes(patchSize: (40, 70), batchSize: batchSize)
        // print("here 1.5")
        let bgBoxes = testData.makeBackgroundBoundingBoxes(patchSize: (40, 70), batchSize: batchSize)
        // print("tests here2")
        let fgpatches = Tensor<Double>(stacking: fgBoxes.map { $0.frame!.patch(at: $0.obb)})
        let bgpatches = Tensor<Double>(stacking: bgBoxes.map { $0.frame!.patch(at: $0.obb)})
        let np = Python.import("numpy")
        let kHiddenDimensions = [512]
        let featSizes = [512]
        print("uu")
        var plt = Python.import("matplotlib.pyplot")
        
        
        for i in featSizes {
        for j in kHiddenDimensions {
        for num in 1...1 {

            let featureSize = i
            let kHiddenDimension = j
            

            let (imageHeight, imageWidth, imageChannels) =
            (40, 70, 1)

            var classifier = NNClassifier(
            imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels,
            hiddenDimension: kHiddenDimension, latentDimension: featureSize
            )
            var weightsFile: String?
            if let weightsFile = weightsFile {
            classifier.load(weights: np.load(weightsFile, allow_pickle: true))
            } else {
            // classifier.load(weights: np.load("./classifiers/classifiers_today/classifier_weight_\(kHiddenDimension)_\(featureSize)_\(num).npy", allow_pickle: true))
            classifier.load(weights: np.load("./classifiers/classifiers_today/classifier_weight_512_512_1_doubletraining.npy", allow_pickle: true))
            }

            let outfg = classifier.classify(fgpatches)
            let outbg = classifier.classify(bgpatches)
            let softmaxfg = softmax(outfg)
            let softmaxbg = softmax(outbg)
            // print(outfg[0...3])
            // print("printing foreground:", softmaxfg[0...10])
            // print("printing background:", softmaxbg[0...10])
            let folderName = "Results/brando06/classified_images"
            if !FileManager.default.fileExists(atPath: folderName) {
            do {
                try FileManager.default.createDirectory(atPath: folderName, withIntermediateDirectories: true, attributes: nil)
            } catch {
                print(error.localizedDescription)
            }
            }
            for i in 0...30 {
                //Background
                var (fig, ax) = plt.subplots(figsize: Python.tuple([8, 4])).tuple2
                var patch = bgpatches[i,0...,0...,0]
                var fr = np.squeeze(patch.makeNumpyArray())
                ax.imshow(fr / 255.0, cmap: "gray")
                ax.set_title("background image: \noutput index 0: \(softmaxbg[i][0])\noutput index 1: \(softmaxbg[i][1])")
                fig.savefig(folderName + "/bgpatch\(i).png", bbox_inches: "tight")
                plt.close("all")
                //Foreground
                (fig, ax) = plt.subplots(figsize: Python.tuple([8, 4])).tuple2
                patch = fgpatches[i,0...,0...,0]
                fr = np.squeeze(patch.makeNumpyArray())
                ax.imshow(fr / 255.0, cmap: "gray")
                ax.set_title("foreground image: \noutput index 0: \(softmaxfg[i][0])\noutput index 1: \(softmaxfg[i][1])")
                fig.savefig(folderName + "/fgpatch\(i).png", bbox_inches: "tight")
                plt.close("all")
            }

            let shapefg = outfg.shape
            let shapebg = outbg.shape
            // print("fg", outfg)
            // print("bg", outbg)

            var fgsum0 = 0.0
            var fgsum1 = 0.0
            var bgsum0 = 0.0
            var bgsum1 = 0.0
            var fg0_arr = [Double]()
            var fg1_arr = [Double]()
            var bg0_arr = [Double]()
            var bg1_arr = [Double]()
            for i in 0...batchSize-1 {
                fgsum0 += Double(softmaxfg[i,0])!
                fgsum1 += Double(softmaxfg[i,1])!
                bgsum0 += Double(softmaxbg[i,0])!
                bgsum1 += Double(softmaxbg[i,1])!
                fg0_arr.append(Double(softmaxfg[i,0])!)
                fg1_arr.append(Double(softmaxfg[i,1])!)
                bg0_arr.append(Double(softmaxbg[i,0])!)
                bg1_arr.append(Double(softmaxbg[i,1])!)
            }
            print("featSize", featureSize, "kHiddendimension", kHiddenDimension, "num", num, "val", fgsum1 + bgsum0 - fgsum0 - bgsum1)

            


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

            figs.savefig("hist_softmax_\(kHiddenDimension)_\(featureSize)_\(num).png")
            plt.close(figs)



        }
        }
        }
        


        

        




    }
}