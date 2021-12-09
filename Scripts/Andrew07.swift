import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation
import ModelSupport

import PenguinStructures

// Error gradient visualization script for Big Transfer
struct Andrew07: ParsableCommand {
    /// This error indicates that BiT-Hyperrule cannot find the name of the dataset in the
    /// knownDatasetSizes dictionary
    enum DatasetNotFoundError: Error {
    case invalidInput(String)
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

    let training = getTrainingDataBigTransfer(from: trainingDataset, numberForeground: 3000, numberBackground: 3000)
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
        //let cropped = tf.image.random_crop(resized.makeNumpyArray(), [batchSize, resizeSize.0, resizeSize.1, 3])
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

    let range = 100.0

    let firstGroundTruth = firstObb.center
    print("oBB coordinates", firstGroundTruth.t.x, firstGroundTruth.t.y)

    let (fig, axs) = plt.subplots(1,2).tuple2
    let fr = np.squeeze(firstFrame.makeNumpyArray())
    axs[0].imshow(fr / 255.0, cmap: "gray")

        
    axs[0].set_xlim(firstGroundTruth.t.x - range/2, firstGroundTruth.t.x + range/2)
    axs[0].set_ylim(firstGroundTruth.t.y - range/2, firstGroundTruth.t.y + range/2)
    axs[1].set_xlim(0, range)
    axs[1].set_ylim(0, range)
    
    let x = firstGroundTruth.t.x
    let y = firstGroundTruth.t.y
    
    var values = Tensor<Double>(zeros: [Int(range), Int(range)])

    for i in 0...Int(range)-1 {
        for j in 0...Int(range)-1 {
            let t = Vector2(x-range/2+Double(i), y-range/2+Double(j))
            let p = Pose2(firstGroundTruth.rot, t)
            var v = VariableAssignments()
            let poseId = v.store(p)
            let startpose = v[poseId]
            var fg = FactorGraph()
            let factorNNC = ProbablisticTrackingFactor2(poseId,
            measurement: firstFrame,
            classifier: bitModel,
            patchSize: (40, 70),
            appearanceModelSize: (40, 70)
            )
            fg.store(factorNNC)
            values[i,j] = Tensor<Double>(factorNNC.errorVector(v[poseId]).x)
            print(j)
            print(i)
        }
    }
    let min_val = values.min()
    if Double(min_val)! < 0 {
    values = values-min_val
    }
    values = values/values.max()*255
    print(values[0...,0])
    print(values.shape)
    axs[1].imshow(values.makeNumpyArray())

    fig.savefig("./Results/andrew01/vizual_NNC.png", bbox_inches: "tight")
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