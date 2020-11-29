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

// MARK: - The Regularized Autoencoder model.

/// A Regularized Autoencoder (RAE) [1] that encodes the appearance of an image patch.
///
/// [1] https://openreview.net/forum?id=S1g7tpEYDS
public struct DenseRAE: Layer {
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
  var encoder_conv1: Conv2D<Double>

  /// Max pooling of factor 2
  var encoder_pool1: MaxPool2D<Double>

  /// First FCN encoding layer goes from image to hidden dimension
  var encoder1: Dense<Double>

  /// Second goes from dense features to latent code
  var encoder2: Dense<Double>

  /// Decode from latent to dense hidden layer with same dimsnions as before
  var decoder1: Dense<Double>

  /// Finally, reconstruct grayscale (or RGB) image
  var decoder2: Dense<Double>

  var decoder_upsample1: UpSampling2D<Double>

  var decoder_conv1: Conv2D<Double>

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
      outputSize: latentDimension)

    decoder1 = Dense<Double>(
      inputSize: latentDimension,
      outputSize: hiddenDimension,
      activation: relu)

    decoder2 = Dense<Double>(
      inputSize: hiddenDimension,
      outputSize: imageHeight * imageWidth * imageChannels / 4)

    decoder_upsample1 = UpSampling2D<Double>(size: 2)

    decoder_conv1 = Conv2D<Double>(filterShape: (3, 3, imageChannels, imageChannels), padding: .same, activation: identity)
  }

  /// Initialize  given an image batch
  public typealias HyperParameters = (hiddenDimension: Int, latentDimension: Int)
  public init(from imageBatch: Tensor<Double>, given parameters: HyperParameters? = nil) {
    let shape = imageBatch.shape
    precondition(imageBatch.rank == 4, "Wrong image shape \(shape)")
    let (_, H_, W_, C_) = (shape[0], shape[1], shape[2], shape[3])
    let (h,d) = parameters ?? (100,10)
    
    var model = DenseRAE(imageHeight: H_, imageWidth: W_, imageChannels: C_,
              hiddenDimension: h, latentDimension: d)
    
    let optimizer = Adam(for: model)
    optimizer.learningRate = 1e-3
    
    let lossFunc = DenseRAELoss()
    
    // Thread-local variable that model layers read to know their mode
    Context.local.learningPhase = .training

    let epochs = TrainingEpochs(samples: imageBatch.unstacked(), batchSize: 200)
    var trainLossResults: [Double] = []
    let epochCount = 200
    for (epochIndex, epoch) in epochs.prefix(epochCount).enumerated() {
      var epochLoss: Double = 0
      var batchCount: Int = 0
      // epoch is a Slices object, see below
      for batchSamples in epoch {
        let batch = batchSamples.collated
        let (loss, grad) = valueWithGradient(at: model) { lossFunc($0, batch) }
        optimizer.update(&model, along: grad)
        epochLoss += loss.scalarized()
        batchCount += 1
      }
      epochLoss /= Double(batchCount)
      trainLossResults.append(epochLoss)
      if epochIndex % 50 == 0 {
          print("Epoch \(epochIndex): Loss: \(epochLoss)")
      }
    }
    
    self = model
  }

  /// Differentiable encoder
  @differentiable(wrt: imageBatch)
  public func encode(_ imageBatch: Tensor<Double>) -> Tensor<Double> {
    let batchSize = imageBatch.shape[0]
    let expectedShape: TensorShape = [batchSize, imageHeight, imageWidth, imageChannels]
    precondition(
        imageBatch.shape == expectedShape,
        "input shape is \(imageBatch.shape), but expected \(expectedShape)")
    return imageBatch
      .sequenced(through: encoder_conv1, encoder_pool1).reshaped(to: [batchSize, imageHeight * imageWidth * imageChannels / 4])
      .sequenced(through: encoder1, encoder2)
  }

  /// Differentiable decoder
  @differentiable
  public func decode(_ latent: Tensor<Double>) -> Tensor<Double> {
    let batchSize = latent.shape[0]
    precondition(
      latent.shape == [batchSize, latentDimension],
      "expected latent shape \([batchSize, latentDimension]), but got \(latent.shape)")
    return decoder_upsample1(latent.sequenced(through: decoder1, decoder2)
      .reshaped(to: [batchSize, imageHeight / 2,  imageWidth / 2, imageChannels]))
  }

  /// Standard: add syntactic sugar to apply model as a function call.
  @differentiable
  public func callAsFunction(_ imageBatch: Tensor<Double>) -> DenseRAEOutput {
    let latent = encode(imageBatch)
    let reconstruction = decode(latent)
    return .init(latent: latent, reconstruction: reconstruction)
  }
}

/// The latent code and reconstructed image that an `DenseRAE` produces given an input image.
public struct DenseRAEOutput: Differentiable {
  /// The latent code for the input image.
  public var latent: Tensor<Double>

  /// The reconstruction of the input image.
  public var reconstruction: Tensor<Double>

  /// Creates an instance with the given `latent` and `reconstruction`.
  @differentiable
  public init(latent: Tensor<Double>, reconstruction: Tensor<Double>) {
    self.latent = latent
    self.reconstruction = reconstruction
  }
}

extension DenseRAE: AppearanceModelEncoder {}

extension DenseRAE {
  /// Jacobian of the `dense` method.
  ///
  /// This is a hand implementation that is much faster than the AD-generated Jacobian.
  public func decodeJacobian(_ latent: Tensor<Double>) -> Tensor<Double> {
    precondition(
      latent.shape == [1, latentDimension],
      "expected latent shape \([1, latentDimension]), but got \(latent.shape)")
    let y_decoder1 = (matmul(latent, decoder1.weight) + decoder1.bias).squeezingShape(at: 0)
    let relu_H_y_decoder1 = Tensor<Double>(y_decoder1 .> Tensor(0)).diagonal()
    // let y2 = matmul(y .> Tensor(0), decoder2.weight)
    let jacobian_decoder2 = matmul(
      decoder2.weight, transposed: true,
      matmul(relu_H_y_decoder1, transposed: false, decoder1.weight, transposed: true))
    let jacobian_reshaped = jacobian_decoder2
      .reshaped(to: [1, imageHeight / 2,  imageWidth / 2, imageChannels * latentDimension])

    let shape = jacobian_reshaped.shape
    let size = 2
    let (batchSize, height, width, channels) = (shape[0], shape[1], shape[2], shape[3])
    let scaleOnes = Tensor<Double>(ones: [1, 1, size, 1, size, 1])
    let upSampling = jacobian_reshaped
      .reshaped(to: [batchSize, height, 1, width, 1, channels]) * scaleOnes

    let jacobian_upscaled = upSampling
      .reshaped(to: [1, imageHeight,  imageWidth, imageChannels, latentDimension])
    return jacobian_upscaled
  }
}

// MARK: - The loss function for the RAE model.

extension DenseRAEOutput {
  // Reconstruction loss is just mean-squared error (defined in s4tf).
  @differentiable
  public func reconstructionLoss(_ data: Tensor<Double>) -> Tensor<Double> {
    return meanSquaredError(predicted: reconstruction, expected: data)
  }

  // Regularization loss is just a unit covariance multivariate Gaussian.
  @differentiable
  public func latentRegularizationLoss() -> Tensor<Double> {
    return latent.squared().mean()
  }
}

extension Dense {
  /// A regularization loss for the parameters of `self`.
  @differentiable
  public func parameterRegularizationLoss() -> Tensor<Scalar> {
    return weight.squared().mean() + bias.squared().mean()
  }
}
extension DenseRAE {
  /// A regularization loss for the parameters of `self`.
  @differentiable
  func parameterRegularizationLoss() -> Tensor<Double> {
    return encoder1.parameterRegularizationLoss() + encoder2.parameterRegularizationLoss()
      + decoder1.parameterRegularizationLoss() + decoder2.parameterRegularizationLoss()
  }
}

/// The loss function for the `DenseRAE`.
public struct DenseRAELoss {
  public let weightReconstruction: Double
  public let weightLatentRegularization: Double
  public let weightParameterRegularization: Double

  /// Creates an instance with default values for the weights of the loss components.
  public init() {
    self.weightReconstruction = 1e0
    self.weightLatentRegularization = 1e-2
    self.weightParameterRegularization = 1e-3
  }

  /// Creates an instance with the given values for the weights of the loss components.
  public init(
    weightReconstruction: Double,
    weightLatentRegularization: Double,
    weightParameterRegularization: Double
  ) {
    self.weightReconstruction = weightReconstruction
    self.weightLatentRegularization = weightLatentRegularization
    self.weightParameterRegularization = weightParameterRegularization
  }

  /// Return the loss of `model` on `imageBatch`.
  ///
  /// Parameter printLoss: Whether to print the loss and its components.
  @differentiable
  public func callAsFunction(
    _ model: DenseRAE, _ imageBatch: Tensor<Double>, printLoss: Bool = false
  ) -> Tensor<Double> {
    let output = model(imageBatch)
    let reconstructionLoss = weightReconstruction * output.reconstructionLoss(imageBatch)
    let latentRegularizationLoss = weightLatentRegularization * output.latentRegularizationLoss()
    let parameterRegularizationLoss = weightParameterRegularization * model.parameterRegularizationLoss()
    let totalLoss = reconstructionLoss + latentRegularizationLoss + parameterRegularizationLoss

    if printLoss {
      print("Reconstruction loss: \(reconstructionLoss)")
      print("Latent regularization loss: \(latentRegularizationLoss)")
      print("Parameter regularization loss: \(parameterRegularizationLoss)")
      print("Total loss: \(totalLoss)")
    }

    return totalLoss
  }
}
