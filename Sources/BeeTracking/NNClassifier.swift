// Copyright 2020 The SwiftFusion Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import SwiftFusion
import TensorFlow
import PythonKit
import BeeDataset

// MARK: - The Regularized Autoencoder model
/// A Regularized Autoencoder (RAE) [1] that encodes the appearance of an image patch.
///

public struct BeeBatch {
  let patch: Tensor<Double>
  let label: Tensor<Int8>
}
/// Conform `IrisBatch` to `Collatable` so that we can load it into a `TrainingEpoch`.
extension BeeBatch: Collatable {
    public init<BatchSamples: Collection>(collating samples: BatchSamples)
        where BatchSamples.Element == Self {
        patch = Tensor<Double>(stacking: samples.map{$0.patch})
        label = Tensor<Int8>(stacking: samples.map{$0.label})
    }
}


/// [1] https://openreview.net/forum?id=S1g7tpEYDS
public struct NNClassifier: Layer{
  /// The height of the input image in pixels.
  @noDerivative public let imageHeight: Int

  /// The width of the input image in pixels.
  @noDerivative public let imageWidth: Int

  /// The number of channels in the input image.
  @noDerivative public let imageChannels: Int

  /// The number of activations in the hidden layer.
  @noDerivative public let hiddenDimension: Int

  /// The number of activations in the appearance code.
  @noDerivative public let latentDimension: Int

  /// First conv to downside the image
  public var encoder_conv1: Conv2D<Double>

  /// Max pooling of factor 2
  var encoder_pool1: MaxPool2D<Double>

  /// First FCN encoding layer goes from image to hidden dimension
  public var encoder1: Dense<Double>

  /// Second goes from dense features to latent code
  public var encoder2: Dense<Double>

  /// Third goes from latent to 1
  public var encoder3: Dense<Double>

  // /// Decode from latent to dense hidden layer with same dimsnions as before
  // var decoder1: Dense<Double>

  // /// Finally, reconstruct grayscale (or RGB) image
  // var decoder2: Dense<Double>

  // var decoder_upsample1: UpSampling2D<Double>

  // var decoder_conv1: Conv2D<Double>

  /// Creates an instance for images with size `[imageHeight, imageWidth, imageChannels]`, with
  /// hidden and latent dimensions given by `hiddenDimension` and `latentDimension`.
  public init(
    imageHeight: Int, imageWidth: Int, imageChannels: Int,
    hiddenDimension: Int, latentDimension: Int
  ) {
    self.imageHeight = imageHeight
    self.imageWidth = imageWidth
    self.imageChannels = imageChannels
    self.hiddenDimension = hiddenDimension
    self.latentDimension = latentDimension

    encoder_conv1 = Conv2D<Double>(filterShape: (3, 3, imageChannels, imageChannels), padding: .same, activation: relu)

    encoder_pool1 = MaxPool2D<Double>(poolSize: (2, 2), strides: (2, 2), padding: .same)
    
    encoder1 = Dense<Double>(
      inputSize: imageHeight * imageWidth * imageChannels / 4,
      outputSize: hiddenDimension,
      activation: relu)

    encoder2 = Dense<Double>(
      inputSize: hiddenDimension,
      outputSize: latentDimension,
      activation: relu)

    encoder3 = Dense<Double>(
      inputSize: latentDimension,
      outputSize: 2)

    }

  /// Initialize  given an image batch
  public typealias HyperParameters = (hiddenDimension: Int, latentDimension: Int, weightFile: String, learningRate: Float)
  // public init(from imageBatch: Tensor<Double>, given parameters: HyperParameters? = nil) {
  public init(patches patches: Tensor<Double>, labels labels: Tensor<Int8>, given parameters: HyperParameters? = nil, train_mode: String) {
    print("init from image batch")
    let (H_, W_, C_) = (patches.shape[1], patches.shape[2], 1)
    let h = parameters!.hiddenDimension
    let d = parameters!.latentDimension
    var model = NNClassifier(imageHeight: H_, imageWidth: W_, imageChannels: C_,
              hiddenDimension: h, latentDimension: d)
    if train_mode == "pretrained" {
      print("PRETRAINED")
      let np = Python.import("numpy")
      print("loading pretrained weights")
      model.load(weights: np.load(parameters!.weightFile, allow_pickle: true))
    }
    
    

    let optimizer = Adam(for: model)
    optimizer.learningRate = parameters!.learningRate
    
    let lossFunc = NNClassifierLoss()
    // Issues I came across: TrainingEpochs function was scrambling the order
    // Then the map function was too slow during training.
    
    // Thread-local variable that model layers read to know their mode
    Context.local.learningPhase = .training

    let trainingData : [BeeBatch] = (zip(patches.unstacked(), labels.unstacked()).map{BeeBatch(patch: $0.0, label: $0.1)})

    let epochs = TrainingEpochs(samples: trainingData, batchSize: 200) // this is an array
    //  
    var trainLossResults: [Double] = []
    let epochCount = 100
    for (epochIndex, epoch) in epochs.prefix(epochCount).enumerated() {
      var epochLoss: Double = 0
      var batchCount: Int = 0
      for batchSamples in epoch {
        let batch = batchSamples.collated
        let (loss, grad) = valueWithGradient(at: model) { lossFunc($0, batch) }
        optimizer.update(&model, along: grad)
        epochLoss += loss.scalarized()
        batchCount += 1
      }
      epochLoss /= Double(batchCount)
      trainLossResults.append(epochLoss)
      if epochIndex % 5 == 0 {
        print("\nEpoch \(epochIndex):", terminator:"")
      }
      print(" \(epochLoss),", terminator: "")
    }
    
    self = model
  }

  /// Differentiable encoder
  @differentiable(wrt: imageBatch)
  public func classify(_ imageBatch: Tensor<Double>) -> Tensor<Double> {
    let batchSize = imageBatch.shape[0]
    let expectedShape: TensorShape = [batchSize, imageHeight, imageWidth, imageChannels]
    precondition(
        imageBatch.shape == expectedShape,
        "input shape is \(imageBatch.shape), but expected \(expectedShape)")
    return imageBatch
      .sequenced(through: encoder_conv1, encoder_pool1).reshaped(to: [batchSize, imageHeight * imageWidth * imageChannels / 4])
      .sequenced(through: encoder1, encoder2, encoder3)
  }

  /// Standard: add syntactic sugar to apply model as a function call.
  @differentiable
  public func callAsFunction(_ imageBatch: Tensor<Double>) -> Tensor<Double> {
    let output = classify(imageBatch)
    return output
  }
}



/// The loss function for the `DenseRAE`.
public struct NNClassifierLoss {

  /// Return the loss of `model` on `imageBatch`.
  /// Parameter printLoss: Whether to print the loss and its components.
  @differentiable
  public func callAsFunction(
    _ model: NNClassifier, _ imageBatch: BeeBatch, printLoss: Bool = false
  ) -> Tensor<Double> {
    let batchSize = imageBatch.patch.shape[0]
    let output = model(imageBatch.patch)
    let totalLoss = softmaxCrossEntropy(logits: output, labels: Tensor<Int32>(imageBatch.label))
    return totalLoss
  }

}

extension NNClassifier: Classifier {}



public struct PretrainedNNClassifier : Classifier{
  public var inner: NNClassifier
  
  /// Constructor that does training of the network
  public init(patches patches: Tensor<Double>, labels labels: Tensor<Int8>, given: HyperParameters, train_mode: String) {
    inner = NNClassifier(
    patches: patches, labels: labels, given: (given != nil) ? 
                                (hiddenDimension: given.hiddenDimension, 
                                latentDimension: given.latentDimension, 
                                weightFile: given.weightFile, 
                                learningRate: given.learningRate) : nil, train_mode: train_mode
    )

    
  }
  
  /// Save the weight to file
  public func save(to path: String) {
    let np = Python.import("numpy")
    np.save(path, np.array(inner.numpyWeights, dtype: Python.object))
  }

  @differentiable
  public func classify(_ imageBatch: Tensor<Double>) -> Tensor<Double> {
    inner.classify(imageBatch)
  }
  
  /// Initialize  given an image batch
  public typealias HyperParameters = (hiddenDimension: Int, latentDimension: Int, weightFile: String, learningRate: Float)
}






















































// /// [1] https://openreview.net/forum?id=S1g7tpEYDS
// public struct SmallerNNClassifier: Layer{
//   @noDerivative public let imageHeight: Int
//   @noDerivative public let imageWidth: Int
//   @noDerivative public let imageChannels: Int
//   @noDerivative public let latentDimension: Int
//   public var encoder_conv1: Conv2D<Double>
//   var encoder_pool1: MaxPool2D<Double>
//   public var encoder1: Dense<Double>
//   public var encoder2: Dense<Double>

//   public init(
//     imageHeight: Int, imageWidth: Int, imageChannels: Int, latentDimension: Int
//   ) {
//     self.imageHeight = imageHeight
//     self.imageWidth = imageWidth
//     self.imageChannels = imageChannels
//     self.latentDimension = latentDimension

//     encoder_conv1 = Conv2D<Double>(filterShape: (3, 3, imageChannels, imageChannels), padding: .same, activation: relu)

//     encoder_pool1 = MaxPool2D<Double>(poolSize: (2, 2), strides: (2, 2), padding: .same)

//     encoder1 = Dense<Double>(
//       inputSize: imageHeight * imageWidth * imageChannels / 4,
//       outputSize: latentDimension,
//       activation: relu)

//     encoder2 = Dense<Double>(
//       inputSize: latentDimension,
//       outputSize: 2)

//     }

//   /// Initialize  given an image batch
//   public init(patches patches: Tensor<Double>, labels labels: Tensor<Int8>, given latentDimension: Int? = nil) {
//     print("init from image batch")
//     let (H_, W_, C_) = (patches.shape[1], patches.shape[2], 1)
//     let d = latentDimension ?? 10
//     var model = SmallerNNClassifier(imageHeight: H_, imageWidth: W_, imageChannels: C_, latentDimension: d)

//     let optimizer = Adam(for: model)
//     optimizer.learningRate = 1e-3
    
//     let lossFunc = NNClassifierLoss()
//     Context.local.learningPhase = .training
//     let trainingData : [BeeBatch] = (zip(patches.unstacked(), labels.unstacked()).map{BeeBatch(patch: $0.0, label: $0.1)})
//     let epochs = TrainingEpochs(samples: trainingData, batchSize: 200) // this is an array
//     var trainLossResults: [Double] = []
//     let epochCount = 600
//     for (epochIndex, epoch) in epochs.prefix(epochCount).enumerated() {
//       var epochLoss: Double = 0
//       var batchCount: Int = 0
//       for batchSamples in epoch {
//         let batch = batchSamples.collated
//         let (loss, grad) = valueWithGradient(at: model) { lossFunc($0, batch) }
//         optimizer.update(&model, along: grad)
//         epochLoss += loss.scalarized()
//         batchCount += 1
//       }
//       epochLoss /= Double(batchCount)
//       trainLossResults.append(epochLoss)
//       // if epochIndex % 50 == 0 {
//       print("Epoch \(epochIndex): Loss: \(epochLoss)")
//       // }
//     }
    
//     self = model
//   }

//   /// Differentiable encoder
//   @differentiable(wrt: imageBatch)
//   public func classify(_ imageBatch: Tensor<Double>) -> Tensor<Double> {
//     let batchSize = imageBatch.shape[0]
//     let expectedShape: TensorShape = [batchSize, imageHeight, imageWidth, imageChannels]
//     precondition(
//         imageBatch.shape == expectedShape,
//         "input shape is \(imageBatch.shape), but expected \(expectedShape)")
//     return imageBatch
//       .sequenced(through: encoder_conv1, encoder_pool1).reshaped(to: [batchSize, imageHeight * imageWidth * imageChannels / 4])
//       .sequenced(through: encoder1, encoder2)
//   }

//   /// Standard: add syntactic sugar to apply model as a function call.
//   @differentiable
//   public func callAsFunction(_ imageBatch: Tensor<Double>) -> Tensor<Double> {
//     let output = classify(imageBatch)
//     return output
//   }
// }

// public struct LargerNNClassifier: Layer{
//   @noDerivative public let imageHeight: Int
//   @noDerivative public let imageWidth: Int
//   @noDerivative public let imageChannels: Int
//   @noDerivative public let hiddenDimension: Int
//   @noDerivative public let latentDimension: Int
//   public var encoder_conv1: Conv2D<Double>
//   var encoder_pool1: MaxPool2D<Double>
//   public var encoder1: Dense<Double>
//   public var encoder2: Dense<Double>
//   public var encoder3: Dense<Double>
//   public var encoder4: Dense<Double>
//   public init(
//     imageHeight: Int, imageWidth: Int, imageChannels: Int,
//     hiddenDimension: Int, latentDimension: Int
//   ) {
//     self.imageHeight = imageHeight
//     self.imageWidth = imageWidth
//     self.imageChannels = imageChannels
//     self.hiddenDimension = hiddenDimension
//     self.latentDimension = latentDimension

//     encoder_conv1 = Conv2D<Double>(filterShape: (3, 3, imageChannels, imageChannels), padding: .same, activation: relu)

//     encoder_pool1 = MaxPool2D<Double>(poolSize: (2, 2), strides: (2, 2), padding: .same)

//     encoder1 = Dense<Double>(
//       inputSize: imageHeight * imageWidth * imageChannels / 4,
//       outputSize: hiddenDimension,
//       activation: relu)

//     encoder2 = Dense<Double>(
//       inputSize: hiddenDimension,
//       outputSize: hiddenDimension,
//       activation: relu)

//     encoder3 = Dense<Double>(
//       inputSize: hiddenDimension,
//       outputSize: latentDimension,
//       activation: relu)

//     encoder4 = Dense<Double>(
//       inputSize: latentDimension,
//       outputSize: 2)

//     }

//   /// Initialize  given an image batch
//   public typealias HyperParameters = (hiddenDimension: Int, latentDimension: Int)
//   // public init(from imageBatch: Tensor<Double>, given parameters: HyperParameters? = nil) {
//   public init(patches patches: Tensor<Double>, labels labels: Tensor<Int8>, given parameters: HyperParameters? = nil) {
//     print("init from image batch")
//     let (H_, W_, C_) = (patches.shape[1], patches.shape[2], 1)
//     let (h,d) = parameters ?? (100,10)
//     var model = LargerNNClassifier(imageHeight: H_, imageWidth: W_, imageChannels: C_,
//               hiddenDimension: h, latentDimension: d)
//     let optimizer = Adam(for: model)
//     optimizer.learningRate = 1e-3
//     let lossFunc = NNClassifierLoss()
//     Context.local.learningPhase = .training
//     let trainingData : [BeeBatch] = (zip(patches.unstacked(), labels.unstacked()).map{BeeBatch(patch: $0.0, label: $0.1)})
//     let epochs = TrainingEpochs(samples: trainingData, batchSize: 200) // this is an array
//     //  
//     var trainLossResults: [Double] = []
//     let epochCount = 600
//     for (epochIndex, epoch) in epochs.prefix(epochCount).enumerated() {
//       var epochLoss: Double = 0
//       var batchCount: Int = 0
//       for batchSamples in epoch {
//         let batch = batchSamples.collated
//         let (loss, grad) = valueWithGradient(at: model) { lossFunc($0, batch) }
//         optimizer.update(&model, along: grad)
//         epochLoss += loss.scalarized()
//         batchCount += 1
//       }
//       epochLoss /= Double(batchCount)
//       trainLossResults.append(epochLoss)
//       if epochIndex % 5 == 0 {
//         print("\nEpoch \(epochIndex):", terminator:"")
//       }
//       print(" \(epochLoss),", terminator: "")
//     }
    
//     // if NSFileManager.fileExistsAtPath(path) {
//     //     print("File exists")
//     // } else {
//     //     print("File does not exist")
//     // }
//     // np.save("epochloss\()", Tensor(trainLossResults).makeNumpyArray())
    
//     self = model
//   }

//   /// Differentiable encoder
//   @differentiable(wrt: imageBatch)
//   public func classify(_ imageBatch: Tensor<Double>) -> Tensor<Double> {
//     let batchSize = imageBatch.shape[0]
//     let expectedShape: TensorShape = [batchSize, imageHeight, imageWidth, imageChannels]
//     precondition(
//         imageBatch.shape == expectedShape,
//         "input shape is \(imageBatch.shape), but expected \(expectedShape)")
//     return imageBatch
//       .sequenced(through: encoder_conv1, encoder_pool1).reshaped(to: [batchSize, imageHeight * imageWidth * imageChannels / 4])
//       .sequenced(through: encoder1, encoder2, encoder3, encoder4)
//   }

//   /// Standard: add syntactic sugar to apply model as a function call.
//   @differentiable
//   public func callAsFunction(_ imageBatch: Tensor<Double>) -> Tensor<Double> {
//     let output = classify(imageBatch)
//     return output
//   }
// }






// public struct PretrainedSmallerNNClassifier : Classifier{
//   public var inner: SmallerNNClassifier
  
//   /// The constructor that only does loading of the pretrained weights.
//   public init(from imageBatch: Tensor<Double>, given: HyperParameters?) {
//     let shape = imageBatch.shape
//     precondition(imageBatch.rank == 4, "Wrong image shape \(shape)")
//     let (_, H_, W_, C_) = (shape[0], shape[1], shape[2], shape[3])
//     if let params = given {
//       var encoder = SmallerNNClassifier(
//         imageHeight: H_, imageWidth: W_, imageChannels: 1, latentDimension: params.latentDimension
//       )

//       let np = Python.import("numpy")

//       encoder.load(weights: np.load(params.weightFile, allow_pickle: true))
//       inner = encoder
//     } else {
//       inner = SmallerNNClassifier(
//         imageHeight: H_, imageWidth: W_, imageChannels: 1, latentDimension: 1
//       )
//       fatalError("Must provide hyperparameters to pretrained network")
//     }
//   }
  
//   /// Constructor that does training of the network
//   public init(patches patches: Tensor<Double>, labels labels: Tensor<Int8>, given: HyperParameters?) {
//     inner = SmallerNNClassifier(
//       patches: patches, labels: labels, given: (given != nil) ? (given!.latentDimension) : nil
//     )
//   }
  
//   /// Save the weight to file
//   public func save(to path: String) {
//     let np = Python.import("numpy")
//     np.save(path, np.array(inner.numpyWeights, dtype: Python.object))
//   }

//   @differentiable
//   public func classify(_ imageBatch: Tensor<Double>) -> Tensor<Double> {
//     inner.classify(imageBatch)
//   }
  
  
//   /// Initialize  given an image batch
//   public typealias HyperParameters = (latentDimension: Int, weightFile: String)
// }






// public struct PretrainedLargerNNClassifier : Classifier{
//   public var inner: LargerNNClassifier
  
//   /// The constructor that only does loading of the pretrained weights.
//   public init(from imageBatch: Tensor<Double>, given: HyperParameters?) {
//     let shape = imageBatch.shape
//     precondition(imageBatch.rank == 4, "Wrong image shape \(shape)")
//     let (_, H_, W_, C_) = (shape[0], shape[1], shape[2], shape[3])
//     if let params = given {
//       var encoder = LargerNNClassifier(
//         imageHeight: H_, imageWidth: W_, imageChannels: 1,
//         hiddenDimension: params.hiddenDimension, latentDimension: params.latentDimension
//       )

//       let np = Python.import("numpy")

//       encoder.load(weights: np.load(params.weightFile, allow_pickle: true))
//       inner = encoder
//     } else {
//       inner = LargerNNClassifier(
//         imageHeight: H_, imageWidth: W_, imageChannels: 1,
//         hiddenDimension: 1, latentDimension: 1
//       )
//       fatalError("Must provide hyperparameters to pretrained network")
//     }
//   }
  
//   /// Constructor that does training of the network
//   public init(patches patches: Tensor<Double>, labels labels: Tensor<Int8>, given: HyperParameters?) {
//     inner = LargerNNClassifier(
//       patches: patches, labels: labels, given: (given != nil) ? (hiddenDimension: given!.hiddenDimension, latentDimension: given!.latentDimension) : nil
//     )
//   }
  
//   /// Save the weight to file
//   public func save(to path: String) {
//     let np = Python.import("numpy")
//     np.save(path, np.array(inner.numpyWeights, dtype: Python.object))
//   }

//   @differentiable
//   public func classify(_ imageBatch: Tensor<Double>) -> Tensor<Double> {
//     inner.classify(imageBatch)
//   }
  
  
//   /// Initialize  given an image batch
//   public typealias HyperParameters = (hiddenDimension: Int, latentDimension: Int, weightFile: String)
// }