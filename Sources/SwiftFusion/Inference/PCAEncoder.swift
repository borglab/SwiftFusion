
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
import TensorFlow
import PenguinStructures

/// A class that does PCA decomposition
/// Input shape for training: [N, H, W, C]
/// W matrix: [H, W, C, latent]
/// Output: [H, W, C]
public struct PCAEncoder {
  public typealias Patch = Tensor<Double>

  /// Basis
  public let U: Tensor<Double>
  
  /// Size of latent
  public var d: Int {
    get {
      U.shape[1]
    }
  }

  /// Input dimension for one sample
  public var n: Int {
    get {
      U.shape[0]
    }
  }

  /// Train a PCAEncoder model
  /// images should be a Tensor of shape [N, H, W, C]
  /// default feature size is 10
  /// Input: [N, H, W, C]
  public typealias HyperParameters = Int
  public init(from imageBatch: Tensor<Double>, given p: HyperParameters? = nil) {
    precondition(imageBatch.rank == 4, "Wrong image shape \(imageBatch.shape)")
    let (N_, H_, W_, C_) = (imageBatch.shape[0], imageBatch.shape[1], imageBatch.shape[2], imageBatch.shape[3])
    let n = H_ * W_ * C_
    let d = p ?? 10

    let images_flattened = imageBatch.reshaped(to: [N_, n]).transposed()
    let (_, U, _) = images_flattened.svd(computeUV: true, fullMatrices: false)

    self.init(withBasis: U![TensorRange.ellipsis, 0..<d])
  }

  /// Initialize a PCAEncoder
  public init(withBasis U: Tensor<Double>) {
    self.U = U
  }

  /// Generate an image according to a latent
  /// Input: [H, W, C]
  /// Output: [d]
  @differentiable
  public func encode(_ image: Patch) -> Tensor<Double> {
    precondition(image.rank == 3 || (image.rank == 4), "wrong latent dimension \(image.shape)")
    let (N_) = (image.shape[0])
    if image.rank == 4 {
      if N_ == 1 {
        return matmul(U, transposed: true, image.reshaped(to: [n, 1])).reshaped(to: [1, d])
      } else {
        let v = image.reshaped(to: [N_, n])
        return matmul(v, U)
      }
    } else {
      return matmul(U, transposed: true, image.reshaped(to: [n, 1])).reshaped(to: [d])
    }
  }
}

extension PCAEncoder: AppearanceModelEncoder {}
