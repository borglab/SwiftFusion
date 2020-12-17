
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

import _Differentiation
import TensorFlow
import PenguinStructures

/// A class performing common activities in the PPCA framework.
/// The decomposition used here is in:
/// M. E. Tipping and C. M. Bishop, Probabilistic Principal Component Analysis,
/// Journal of the Royal Statistical Society. Series B, Vol. 61, No. 3 (1999), pp. 611-622
/// Input shape for training: [N, H, W, C]
/// W matrix: [H, W, C, latent]
/// Output: [H, W, C]
public struct PPCA {
  public typealias Patch = Tensor<Double>

  /// Weight Matrix
  public var W: Tensor<Double>
  
  /// Inverse of Weight Matrix
  public var W_inv: Tensor<Double>?

  /// Bias
  public var mu: Patch

  /// Size of latent
  public var latent_size: Int

  /// Basis
  public var Ut: Tensor<Double>?

  /// Constructor
  /// measured: Measured Patch (Template)
  /// W and mu are calculated by PPCA
  public init(W: Tensor<Double>, mu: Patch) {
    self.latent_size = W.shape[3]
    self.W = W
    self.mu = mu
    self.W_inv = nil
    self.Ut = nil
  }

  public init(latentSize: Int) {
    self.W = Tensor([.nan])
    self.W_inv = nil
    self.mu = Tensor([.nan])
    self.latent_size = latentSize
    self.Ut = nil
  }

  /// Initialize  given an image batch
  public typealias HyperParameters = Int
  public init(from imageBatch: Tensor<Double>, given d: HyperParameters? = nil) {
    self.init(latentSize: d ?? 5)
    train(images: imageBatch)
  }

  /// Train a PPCA model
  /// images should be a Tensor of shape [N, H, W, C]
  /// Input: [N, H, W, C]
  public mutating func train(images: Tensor<Double>) {
    let shape = images.shape
    precondition(images.rank == 4, "Wrong image shape \(shape)")
    let (N_, H_, W_, C_) = (shape[0], shape[1], shape[2], shape[3])
    
    self.mu = images.mean(squeezingAxes: [0])
    let d = H_ * W_ * C_
    let images_flattened = (images - mu).reshaped(to: [N_, d]).transposed()
    /// U.shape should be [d, rank]
    let (S, U, _) = images_flattened.svd(computeUV: true, fullMatrices: false)

    let sigma_2 = S[latent_size...].mean()
    
    self.Ut = U![TensorRange.ellipsis, 0..<latent_size].transposed()

    self.W = matmul(
      self.Ut!.transposed(),
      (S[0..<latent_size] - sigma_2).diagonal()
    ).reshaped(to: [H_, W_, C_, latent_size])

    // TODO: Cache A^TA?
    if self.W_inv == nil {
      // self.W_inv = pinv(W.reshaped(to: [d, latent_size]))
      let W_m = W.reshaped(to: [d, latent_size])
      self.W_inv = matmul(pinv(matmul(W_m.transposed(), W_m)), W_m.transposed())
    }
  }

  /// Generate an image according to a latent
  /// Input: [latent_size] or [latent_size, 1]
  @differentiable
  public func decode(_ latent: Tensor<Double>) -> Patch {
    precondition(latent.rank == 1 || (latent.rank == 2 && latent.shape[1] == 1), "wrong latent dimension \(latent.shape)")
    if(latent.rank == 1) {
      return mu + matmul(W, latent.expandingShape(at: [1])).squeezingShape(at: 3)
    }
    return mu + matmul(W, latent).squeezingShape(at: 3)
  }

  /// Generate an image according to a latent
  /// Input: [H, W, C]
  /// Output: [latent_size]
  @differentiable
  public func encode(_ image: Patch) -> Tensor<Double> {
    precondition(image.rank == 3 || (image.rank == 4), "wrong latent dimension \(image.shape)")
    let (N_, H_, W_, C_) = (image.shape[0], W.shape[0], W.shape[1], W.shape[2])
    if image.rank == 4 {
      if N_ == 1 {
        return matmul(W_inv!, (image - mu).reshaped(to: [H_ * W_ * C_, 1])).reshaped(to: [1, latent_size])
      } else {
        let v_T = (image - mu).reshaped(to: [H_ * W_ * C_, N_]).transposed()
        return matmul(v_T, W_inv!.transposed()).reshaped(to: [N_, latent_size])
      }
    } else {
      return matmul(W_inv!, (image - mu).reshaped(to: [H_ * W_ * C_, 1])).reshaped(to: [latent_size])
    }
  }

  /// Generate an image and corresponding Jacobian according to a latent
  /// Input: [latent_size] or [latent_size, 1]
  public func decodeWithJacobian(_ latent: Tensor<Double>) -> (Patch, Patch.TangentVector) {
    return (decode(latent), W)
  }
}

extension PPCA: AppearanceModelEncoder {}
