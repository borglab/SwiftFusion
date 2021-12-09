import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation


import PenguinStructures

// PRINT IMAGE PATCHES TO VISUALIZE
struct Brando08: ParsableCommand {    

    func run() {
        let dataDir = URL(fileURLWithPath: "./OIST_Data")
        let dataset = OISTBeeVideo(directory: dataDir, length: 100)!
        let batchSize = 300
        let fgBoxes = dataset.makeForegroundBoundingBoxes(patchSize: (40, 70), batchSize: batchSize)
        let bgBoxes = dataset.makeBackgroundBoundingBoxes(patchSize: (40, 70), batchSize: batchSize)
        let fgpatches = Tensor<Double>(stacking: fgBoxes.map { $0.frame!.patch(at: $0.obb)})
        let bgpatches = Tensor<Double>(stacking: bgBoxes.map { $0.frame!.patch(at: $0.obb)})
        let np = Python.import("numpy")
        var plt = Python.import("matplotlib.pyplot")
        let mpl = Python.import("matplotlib")

        print(fgpatches.shape)
        for i in batchSize-100...batchSize-1 {
            let (fig, ax) = plt.subplots(figsize: Python.tuple([8, 4])).tuple2
            let patch = bgpatches[i,0...,0...,0]
            let fr = np.squeeze(patch.makeNumpyArray())
            ax.imshow(fr / 255.0, cmap: "gray")
            let folderName = "Results/brando08/bgpatches"
            if !FileManager.default.fileExists(atPath: folderName) {
            do {
                try FileManager.default.createDirectory(atPath: folderName, withIntermediateDirectories: true, attributes: nil)
            } catch {
                print(error.localizedDescription)
            }
            }
            fig.savefig("Results/brando08/bgpatches/patch\(i).png", bbox_inches: "tight")
            plt.close("all")
            
        }
    }
}
