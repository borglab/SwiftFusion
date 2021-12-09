import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation
import ModelSupport

import PenguinStructures

let tf = Python.import("tensorflow")
let np  = Python.import("numpy")
let pickle = Python.import("pickle")

// Optional to enable GPU training
// let _ = _ExecutionContext.global
// let device = Device.defaultXLA
let device = Device.default
let modelName = "BiT-M-R50x1"
var knownModels = [String: String]()
let knownDatasetSizes:[String: (Int, Int)] = [
  "bee_dataset": (40, 70)
]

public struct LabeledData<Data, Label> {
  /// The `data` of our sample (usually used as input for a model).
  public let data: Data
  /// The `label` of our sample (usually used as target for a model).
  public let label: Label

  /// Creates an instance from `data` and `label`.
  public init(data: Data, label: Label) {
    self.data = data
    self.label = label
  }
}

// Script to train and track with Big Transfer
struct Andrew06: ParsableCommand {
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
    let dataDir = URL(fileURLWithPath: "./OIST_Data")

    let trainingDataset = OISTBeeVideo(directory: dataDir, length: 80)!
    let validationDataset = OISTBeeVideo(directory: dataDir, afterIndex: 80, length: 20)!

    let training = getTrainingDataBigTransfer(from: trainingDataset, numberForeground: 20000, numberBackground: 20000)
    let validation = getTrainingDataBigTransfer(from: validationDataset, numberForeground: 600, numberBackground: 600)

    
    let classCount = 2
    var bitModel = BigTransfer(classCount: classCount, depth: getModelUnits(modelName: modelName), modelName: modelName)
    let dataCount = 40000

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


    
    let evalTracker: Tracker = {frames, start in
        var tracker = makeProbabilisticTracker2(
            model: bitModel,
            frames: frames,
            targetSize: (40, 70)
        )
        let prediction = tracker.infer(knownStart: Tuple1(start.center), withSampling: true)
        let track = tracker.frameVariableIDs.map { OrientedBoundingBox(center: prediction[unpack($0)], rows: 40, cols:70) }
        return track

    }

    let plt = Python.import("matplotlib.pyplot")
    let sequenceCount = 19
    var results = trackerEvaluation.evaluate(evalTracker, sequenceCount: sequenceCount, deltaAnchor: 175, outputFile: "andrew01")

    for (index, value) in results.sequences.prefix(sequenceCount).enumerated() {
      var i: Int = 0
      zip(value.subsequences.first!.frames, zip(value.subsequences.first!.prediction, value.subsequences.first!.groundTruth)).map {
        let (fig, axes) = plotFrameWithPatches(frame: $0.0, actual: $0.1.0.center, expected: $0.1.1.center, firstGroundTruth: value.subsequences.first!.groundTruth.first!.center)
        fig.savefig("Results/andrew01/sequence\(index)/andrew01_\(i).png", bbox_inches: "tight")
        plt.close("all")
        i = i + 1
      }
      
      
      let (fig, axes) = plt.subplots(1, 2, figsize: Python.tuple([20, 20])).tuple2
      fig.suptitle("Tracking positions and Subsequence Average Overlap with Accuracy \(String(format: "%.2f", value.subsequences.first!.metrics.accuracy)) and Robustness \(value.subsequences.first!.metrics.robustness).")
      
      value.subsequences.map {
        let encoder = JSONEncoder()
        let data = try! encoder.encode($0.prediction)
        FileManager.default.createFile(atPath: "prediction_bigtransfer_sequence_\(index).json", contents: data, attributes: nil)
        plotPoseDifference(
          track: $0.prediction.map{$0.center}, withGroundTruth: $0.groundTruth.map{$0.center}, on: axes[0]
        )
      }
      plotOverlap(
          metrics: value.subsequences.first!.metrics, on: axes[1]
      )
      fig.savefig("Results/andrew01/andrew01_subsequence\(index).png", bbox_inches: "tight")
      print("Accuracy for sequence is \(value.sequenceMetrics.accuracy) with Robustness of \(value.sequenceMetrics.robustness)")
    }

    print("Accuracy for all sequences is \(results.trackerMetrics.accuracy) with Robustness of \(results.trackerMetrics.robustness)")
    let f = Python.open("Results/EAO/bigtransfer.data", "wb")
    pickle.dump(results.expectedAverageOverlap.curve, f)


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