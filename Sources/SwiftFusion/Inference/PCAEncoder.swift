
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

  /// Size of latent
  public var latent_size: Int

  /// Basis
  public var U: Tensor<Double>?

  public init(latentSize: Int) {
    self.latent_size = latentSize
    self.U = nil
  }

  /// Train a PCAEncoder model
  /// images should be a Tensor of shape [N, H, W, C]
  /// Input: [N, H, W, C]
  public mutating func train(images: Tensor<Double>) {
    precondition(images.rank == 4, "Wrong image shape \(images.shape)")
    let (N_, H_, W_, C_) = (images.shape[0], images.shape[1], images.shape[2], images.shape[3])
    
    let images_flattened = images.reshaped(to: [N_, H_ * W_ * C_]).transposed()
    let (_, J_u, _) = images_flattened.svd(computeUV: true, fullMatrices: false)

    self.U = J_u![
        0..<J_u!.shape[0],
        0..<latent_size
    ].reshaped(to: [H_, W_, C_, latent_size])
  }

  /// Generate an image according to a latent
  /// Input: [H, W, C]
  /// Output: [latent_size]
  @differentiable
  public func encode(_ image: Patch) -> Tensor<Double> {
    precondition(image.rank == 3 || (image.rank == 4), "wrong latent dimension \(image.shape)")
    let (N_, H_, W_, C_) = (image.shape[0], U!.shape[0], U!.shape[1], U!.shape[2])
    let d = H_ * W_ * C_
    if image.rank == 4 {
      if N_ == 1 {
        return matmul(U!.reshaped(to: [d, latent_size]), transposed: true, image.reshaped(to: [d, 1])).reshaped(to: [1, latent_size])
      } else {
        let v_T = image.reshaped(to: [d, N_]).transposed()
        return matmul(v_T, U!.reshaped(to: [d, latent_size])).reshaped(to: [N_, latent_size])
      }
    } else {
      return matmul(U!.reshaped(to: [d, latent_size]), transposed: true, image.reshaped(to: [d, 1])).reshaped(to: [latent_size])
    }
  }
}

extension PCAEncoder: AppearanceModelEncoder {}
