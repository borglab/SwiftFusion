import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation
import ModelSupport

import PenguinStructures


struct Andrew08: ParsableCommand {
    /// This error indicates that BiT-Hyperrule cannot find the name of the dataset in the
    /// knownDatasetSizes dictionary
    enum DatasetNotFoundError: Error {
    case invalidInput(String)
    }
    func initialize_and_perturb(p: Pose2) -> (Double, Double, Double, Pose2, VariableAssignments, TypedID<Pose2>, FactorGraph) {
        var v = VariableAssignments()
        let poseId = v.store(p)
        v[poseId].perturbWith(stddev: Vector3(0.3, 8, 4.6))
        let dx = v[poseId].t.x - p.t.x
        let dy = v[poseId].t.y - p.t.y
        let dtheta = v[poseId].rot.theta - p.rot.theta
        let startpose = v[poseId]
        let fg = FactorGraph()

        return (dx, dy, dtheta, startpose, v, poseId, fg)
    }

    func initialize_empty_arrays() -> (Bool, [Double], [Double], [Double], [Double]) {
        var conv = true
        var errors = [Double]()
        var xs = [Double]()
        var ys = [Double]()
        var thetas = [Double]()
        return (conv, errors, xs, ys, thetas)
    }
    /// Return relevent ResNet enumerated type based on weights loaded
    ///
    /// - Parameters:
    ///   - modelName: the name of the model pulled from the big transfer repository
    ///                to grab the enumerated type for
    /// - Returns: ResNet enumerated type for BigTransfer model
    func getModelUnits(modelName: String) -> BigTransfer.Depth {
    if modelName.contains("R50") {
        return .resNet50
    }
    else if modelName.contains("R101") {
        return .resNet101
    }
    else {
        return .resNet152
    }
    }

    /// Get updated image resolution based on the specifications in BiT-Hyperrule
    ///
    /// - Parameters:
    ///   - originalResolution: the source resolution for the current image dataset
    /// - Returns: new resolution for images based on BiT-Hyperrule
    func getResolution(originalResolution: (Int, Int)) -> (Int, Int) {
    let area = originalResolution.0 * originalResolution.1
    return area < 96*96 ? (160, 128) : (512, 480)
    }

    /// Get the source resolution for the current image dataset from the knownDatasetSizes dictionary
    ///
    /// - Parameters:
    ///   - datasetName: name of the current dataset you are using
    /// - Returns: new resolution for specified dataset
    /// - Throws:
    ///   - DatasetNotFoundError: will throw an error if the dataset cannot be found in knownDatasetSizes dictionary
    func getResolutionFromDataset(datasetName: String) throws -> (Int, Int) {
    if let resolution = knownDatasetSizes[datasetName] {
        return getResolution(originalResolution: resolution)
    }
    print("Unsupported dataset " + datasetName + ". Add your own here :)")
    throw DatasetNotFoundError.invalidInput(datasetName)

    }

    /// Get training mixup parameters based on Bit-Hyperrule specification for dataset sizes
    ///
    /// - Parameters:
    ///   - datasetSize: number of images in the current dataset
    /// - Returns: mixup alpha based on number of images
    func getMixUp(datasetSize: Int) -> Double {
    return datasetSize < 20000 ? 0.0 : 0.1
    }

    /// Get the learning rate schedule based on the dataset size
    ///
    /// - Parameters:
    ///   - datasetSize: number of images in the current dataset
    /// - Returns: learning rate schedule based on the current dataset
    func getSchedule(datasetSize: Int) -> Array<Int> {
    if datasetSize == 100 {
        return [25, 50, 75, 100]
    }
    if datasetSize < 20000{
        return [100, 200, 300, 400, 500]
    }
    else if datasetSize < 500000 {
        return [500, 3000, 6000, 9000, 10000]
    }
    else {
        return [500, 6000, 12000, 18000, 20000]
    }
    }

    /// Get learning rate at the current step given the dataset size and base learning rate
    ///
    /// - Parameters:
    ///   - step: current training step
    ///   - datasetSize: number of images in the dataset
    ///   - baseLearningRate: starting learning rate to modify
    /// - Returns: learning rate at the current step in training
    func getLearningRate(step: Int, datasetSize: Int, baseLearningRate: Float = 0.003) -> Float? {
    let supports = getSchedule(datasetSize: datasetSize)
    // Linear warmup
    print(step)
    print(supports)
    if step < supports[0] {
        return baseLearningRate * Float(step) / Float(supports[0])
    }
    // End of training
    else if step >= supports.last! {
        return nil
    }
    // Staircase decays by factor of 10
    else {
        var baseLearningRate = baseLearningRate
        for s in supports[1...] {
        if s < step {
            baseLearningRate = baseLearningRate / 10.0
        }
        }
        return baseLearningRate
    }
 }
  public typealias Datum = (patch: Tensor<Float>, label: Tensor<Int32>)
  public typealias LabeledImage = LabeledData<Tensor<Float>, Tensor<Int32>>
  public typealias Batches = Slices<Sampling<[(patch: Tensor<Float>, label: Tensor<Int32>)], ArraySlice<Int>>>

  func getTrainingDataBigTransfer(
    from dataset: OISTBeeVideo,
    numberForeground: Int = 10000,
    numberBackground: Int = 10000
    ) -> [Datum] {
    let bgBoxes = dataset.makeBackgroundBoundingBoxes(patchSize: (40, 70), batchSize: numberBackground).map {
      (patch: Tensor<Float>($0.frame!.patch(at: $0.obb)).unstacked(alongAxis: 2)[0], label: Tensor<Int32>(0))
    }
    let fgBoxes = dataset.makeForegroundBoundingBoxes(patchSize: (40, 70), batchSize: numberForeground).map {
      (patch: Tensor<Float>($0.frame!.patch(at: $0.obb)).unstacked(alongAxis: 2)[0], label: Tensor<Int32>(1))
    }
    
    var boxes = fgBoxes + bgBoxes
    return boxes.map{(patch: Tensor<Float>(stacking: [$0.patch, $0.patch, $0.patch], alongAxis: 2), label: $0.label)}
  }
  
    /// Stores the training statistics for the BigTransfer training process which are different than usual
    /// because the mixedup labels must be accounted for while running training statistics.
    struct BigTransferTrainingStatistics {
        var correctGuessCount = Tensor<Int32>(0, on: Device.default)
        var totalGuessCount = Tensor<Int32>(0, on: Device.default)
        var totalLoss = Tensor<Float>(0, on: Device.default)
        var batches: Int = 0
        var accuracy: Float { 
            Float(correctGuessCount.scalarized()) / Float(totalGuessCount.scalarized()) * 100 
        } 
        var averageLoss: Float { totalLoss.scalarized() / Float(batches) }

        init(on device: Device = Device.default) {
            correctGuessCount = Tensor<Int32>(0, on: device)
            totalGuessCount = Tensor<Int32>(0, on: device)
            totalLoss = Tensor<Float>(0, on: device)
        }

        mutating func update(logits: Tensor<Float>, labels: Tensor<Float>, loss: Tensor<Float>) {
            let correct = logits.argmax(squeezingAxis: 1) .== labels.argmax(squeezingAxis: 1)
            correctGuessCount += Tensor<Int32>(correct).sum()
            totalGuessCount += Int32(labels.shape[0])
            totalLoss += loss
            batches += 1
        }
    }

  fileprivate func makeBatch<BatchSamples: Collection>(
    samples: BatchSamples, device: Device) -> LabeledImage where BatchSamples.Element == (patch: Tensor<Float>, label: Tensor<Int32>) {
    let labels = Tensor<Int32>(samples.map(\.label))
    let imageTensor = Tensor<Float>(samples.map(\.patch))
    return LabeledImage(data: imageTensor, label: labels)
}
  // Train Big Transfer
  func run() {
    let plt = Python.import("matplotlib.pyplot")
    let dataDir = URL(fileURLWithPath: "./OIST_Data")

    let trainingDataset = OISTBeeVideo(directory: dataDir, length: 80)!
    let validationDataset = OISTBeeVideo(directory: dataDir, afterIndex: 80, length: 20)!

    let training = getTrainingDataBigTransfer(from: trainingDataset, numberForeground: 256, numberBackground: 256)
    let validation = getTrainingDataBigTransfer(from: validationDataset, numberForeground: 600, numberBackground: 600)

    
    let classCount = 2
    var bitModel = BigTransfer(classCount: classCount, depth: getModelUnits(modelName: modelName), modelName: modelName)
    let dataCount = 6000

    var optimizer = SGD(for: bitModel, learningRate: 0.003, momentum: 0.9)
    optimizer = SGD(copying: optimizer, to: device)

    print("Beginning training...")
    var batchSize: Int = 16
    var currStep: Int = 1
    let lrSupports = getSchedule(datasetSize: dataCount)
    let scheduleLength = lrSupports.last!
    let stepsPerEpoch = dataCount / batchSize
    let epochCount = scheduleLength / stepsPerEpoch
    let resizeSize = getResolution(originalResolution: (40, 70))

    let trainingData = TrainingEpochs(samples: training, batchSize: batchSize).lazy.map { 
        (batches: Batches) -> LazyMapSequence<Batches, LabeledImage> in
            return batches.lazy.map{ makeBatch(samples: $0, device: device) } 
        }

    let validationData = validation.inBatches(of: batchSize).lazy.map {
      makeBatch(samples: $0, device: device)
    }

    for (epoch, batches) in trainingData.prefix(epochCount).enumerated() {
        let start = Date()
        var trainStats = BigTransferTrainingStatistics(on: device)
        var testStats = BigTransferTrainingStatistics(on: device)

        Context.local.learningPhase = .training
        for batch in batches {
        if let newLearningRate = getLearningRate(step: currStep, datasetSize: dataCount, baseLearningRate: 0.003) {
            optimizer.learningRate = newLearningRate
            currStep = currStep + 1
        }
        else {
            continue
        }

        var (eagerImages, eagerLabels) = (batch.data, batch.label)
        let resized = resize(images: eagerImages, size: (resizeSize.0, resizeSize.1))
        let flipped = tf.image.random_flip_left_right(resized.makeNumpyArray())
        var newLabels = Tensor<Float>(Tensor<Int32>(oneHotAtIndices: eagerLabels, depth: classCount))
        
        let images = Tensor(copying: Tensor<Float>(numpy: flipped.numpy())!, to: device)
        let labels = Tensor(copying: newLabels, to: device)
        let 𝛁model = TensorFlow.gradient(at: bitModel) { bitModel -> Tensor<Float> in
            let ŷ = bitModel(images)
            let loss = softmaxCrossEntropy(logits: ŷ, probabilities: labels)
            trainStats.update(logits: ŷ, labels: labels, loss: loss)
            return loss
        }

        optimizer.update(&bitModel, along: 𝛁model)
        
        LazyTensorBarrier()
        }

        print("Checking validation statistics...")
        Context.local.learningPhase = .inference
        for batch in validationData {
            var (eagerImages, eagerLabels) = (batch.data, batch.label)
            let resized = resize(images: eagerImages, size: (resizeSize.0, resizeSize.1))
            let newLabels = Tensor<Float>(Tensor<Int32>(oneHotAtIndices: eagerLabels, depth: classCount))
            let images = Tensor(copying: resized, to: device)
            let labels = Tensor(copying: newLabels, to: device)
            let ŷ = bitModel(images)
            let loss = softmaxCrossEntropy(logits: ŷ, probabilities: labels)
            LazyTensorBarrier()
            testStats.update(logits: ŷ, labels: labels, loss: loss)
        }

        print(
            """
            [Epoch \(epoch)] \
            Training Loss: \(String(format: "%.3f", trainStats.averageLoss)), \
            Training Accuracy: \(trainStats.correctGuessCount)/\(trainStats.totalGuessCount) \
            (\(String(format: "%.1f", trainStats.accuracy))%), \
            Test Loss: \(String(format: "%.3f", testStats.averageLoss)), \
            Test Accuracy: \(testStats.correctGuessCount)/\(testStats.totalGuessCount) \
            (\(String(format: "%.1f", testStats.accuracy))%) \
            seconds per epoch: \(String(format: "%.1f", Date().timeIntervalSince(start)))
            """)
    }

    let testData = OISTBeeVideo(directory: dataDir, afterIndex: 100, length: 80)!

    let trackerEvaluation = TrackerEvaluationDataset(testData) 

    let frames = testData.frames
    let firstTrack = testData.tracks[0]
    let firstFrame = frames[0]
    let firstObb = firstTrack.boxes[0]

    let lr = 100.0
    var GDOptimizer = GradientDescent(learningRate: lr)
    let it_limit = 80


    let folderName = "Results/GD_optimization_BiT_lr_\(lr)__10_22_2021_final_images_4subplots"
    if !FileManager.default.fileExists(atPath: folderName) {
    do {
        try FileManager.default.createDirectory(atPath: folderName, withIntermediateDirectories: true, attributes: nil)
    } catch {
        print(error.localizedDescription)
    }
    }

    print("hello1")
    let (fig, axs) = plt.subplots(2,2).tuple2
    let fr = np.squeeze(firstFrame.makeNumpyArray())
    for i in 0...1 {
      for j in 0...1 {
        axs[i,j].imshow(fr / 255.0, cmap: "gray")
        let firstGroundTruth = firstObb.center
        axs[i,j].set_xlim(firstGroundTruth.t.x - 50, firstGroundTruth.t.x + 50)
        axs[i,j].set_ylim(firstGroundTruth.t.y - 50, firstGroundTruth.t.y + 50)
        axs[i,j].get_xaxis().set_visible(false)
        axs[i,j].get_yaxis().set_visible(false)
      }
    }
    axs[0,0].set_title("fabs(theta) < 6 Degrees", fontsize:8)
    axs[0,1].set_title("fabs(theta) < 12 Degrees", fontsize:8)
    axs[1,0].set_title("fabs(theta) < 16 Degrees", fontsize:8)
    axs[1,1].set_title("fabs(theta) >= 16 Degrees", fontsize:8)

    print("hello")
    let xy_thresh = 20.0 //pixels
    let theta_thresh = 0.5 //radians // consider doing overlap.

    
    // NN Params
    let (imageHeight, imageWidth, imageChannels) = (40, 70, 1)
    let featureSize = 256
    let kHiddenDimension = 512

    var useClassifier = true
    if useClassifier {
      var classifier = bitModel
      for j in 0...200 {
        var (dx, dy, dtheta, startpose, v, poseId, fg) = initialize_and_perturb(p: firstObb.center)
        let factorNNC = ProbablisticTrackingFactor2(poseId,
          measurement: firstFrame,
          classifier: classifier,
          patchSize: (40, 70),
          appearanceModelSize: (40, 70)
        )
        fg.store(factorNNC)


        // PERFORM GRADIENT DESCENT
        var (conv, errors, xs, ys, thetas) = initialize_empty_arrays()
        print("starting optimization")
        for i in 0..<it_limit {
            errors.append(factorNNC.errorVector(v[poseId]).x)
            xs.append(v[poseId].t.x)
            ys.append(v[poseId].t.y)
            thetas.append(v[poseId].rot.theta)
            let oldpose = v[poseId]
            GDOptimizer.learningRate = Double(getLearningRate(step: i + 1, datasetSize: 100, baseLearningRate: 0.01)!)
            GDOptimizer.update(&v, objective: fg)

            print(v[poseId].t.x - oldpose.t.x)
            if abs(v[poseId].t.x - oldpose.t.x) < 0.000001 && abs(v[poseId].t.y - oldpose.t.y) < 0.000001 && abs(v[poseId].rot.theta - oldpose.rot.theta) < 0.000001{
            break
            }
            if i == it_limit-1 {
            conv = false
            }
        }

        let x_out_of_bounds = (v[poseId].t.x > firstObb.center.t.x + xy_thresh) || (v[poseId].t.x < firstObb.center.t.x - xy_thresh)
        let y_out_of_bounds =  (v[poseId].t.y > firstObb.center.t.y + xy_thresh) || (v[poseId].t.y < firstObb.center.t.y - xy_thresh)
        let theta_out_of_bounds = (v[poseId].rot.theta > firstObb.center.rot.theta + theta_thresh) || (v[poseId].rot.theta < firstObb.center.rot.theta - theta_thresh)
        if !x_out_of_bounds && !theta_out_of_bounds && !y_out_of_bounds {
            if fabs(startpose.rot.theta - firstObb.center.rot.theta) < 0.1 {
                axs[0,0].plot(startpose.t.x,startpose.t.y,"g,", ms: 5)
            } else if fabs(startpose.rot.theta - firstObb.center.rot.theta) < 0.2 {
                axs[0,1].plot(startpose.t.x,startpose.t.y,"g,", ms: 5)
            } else if fabs(startpose.rot.theta - firstObb.center.rot.theta) < 0.3 {
                axs[1,0].plot(startpose.t.x,startpose.t.y,"g,", ms: 5)
            } else {
                axs[1,1].plot(startpose.t.x,startpose.t.y,"g,", ms: 5)
            }
            
        } else {
            if fabs(startpose.rot.theta - firstObb.center.rot.theta) < 0.1 {
                axs[0,0].plot(startpose.t.x,startpose.t.y,"r,", ms: 5)
            } else if fabs(startpose.rot.theta - firstObb.center.rot.theta) < 0.2 {
                axs[0,1].plot(startpose.t.x,startpose.t.y,"r,", ms: 5)
            } else if fabs(startpose.rot.theta - firstObb.center.rot.theta) < 0.3 {
                axs[1,0].plot(startpose.t.x,startpose.t.y,"r,", ms: 5)
            } else {
                axs[1,1].plot(startpose.t.x,startpose.t.y,"r,", ms: 5)
            }
        }
        let (figs, axes) = plotFrameWithPatches3(frame: firstFrame, start: startpose, end: v[poseId], expected: firstObb.center, firstGroundTruth: firstObb.center, errors: errors, xs: xs, ys: ys, thetas: thetas)
        var final_err: Double
        var label_err: Double
        var start_err: Double

        
        final_err = factorNNC.errorVector(v[poseId]).x
        label_err = factorNNC.errorVector(firstObb.center).x
        start_err = factorNNC.errorVector(startpose).x

        axes.set_title(String(axes.get_title())! + "\n final err = \(final_err)" 
        + "\n label err = \(label_err).x)" 
        + "\n start err = \(start_err)"
        + "\n learning rate = \(lr)"
        + "\n converged = \(conv)")
        figs.savefig(folderName + "/optimization_final_\(j).png", bbox_inches: "tight")
        plt.close("all")
        fig.savefig(folderName + "/optimization_covergence_red_n_green_dots.png", bbox_inches: "tight")
    
        }
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