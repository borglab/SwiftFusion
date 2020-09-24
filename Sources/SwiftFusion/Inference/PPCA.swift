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

/// A factor that matches a patch on a latent variable.
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
  
  /// Bias
  public var mu: Patch

  /// Size of latent
  public var latent_size: Int

  /// Constructor
  /// measured: Measured Patch (Template)
  /// W and mu are calculated by PPCA
  public init(W: Tensor<Double>, mu: Patch) {
    self.latent_size = W.shape[3]
    self.W = W
    self.mu = mu
  }

  public init(latentSize: Int) {
    self.W = .init()
    self.mu = .init()
    self.latent_size = latentSize
  }

  /// [N, H, W, C]
  public mutating func train(images: Tensor<Double>) {
    precondition(images.rank == 4, "Wrong image shape")
    let (N_, H_, W_, C_) = (images.shape[0], images.shape[1], images.shape[2], images.shape[3])
    
    self.mu = images.mean(alongAxes: [0])
    let images_flattened = (images - mu).reshaped(to: [N_, H_ * W_ * C_]).transposed()
    let (J_s, J_u, _) = images_flattened.svd(computeUV: true, fullMatrices: false)
    
    let sigma_2 = J_s[latent_size...].mean()
    self.W = matmul(J_u![0..<J_u!.shape[0], 0..<latent_size], (J_s[0..<latent_size] - sigma_2).diagonal()).reshaped(to: [H_, W_, C_, latent_size])
  }

  /// Error Vector for PPCA Factor
  /// e = W*a + mu - measured
  @differentiable
  public func generate(_ latent: Tensor<Double>) -> Patch {
    precondition(latent.rank == 2 && latent.shape[1] == 1, "wrong latent dimension")
    return mu + matmul(W, latent).squeezingShape(at: 3)
  }
}
