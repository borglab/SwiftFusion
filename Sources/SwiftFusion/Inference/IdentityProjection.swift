
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

/// A class performing common activities in the IdentityProjection framework.
/// 
/// NOTE: This class can do normalization
/// 
/// - Input shape for training: [N, H, W, C]
/// - W matrix: [feature, H*W*C]
/// - Output: [feature]
public struct IdentityProjection {
  public typealias Patch = Tensor<Double>

  /// Sample mean
  public let mean: Tensor<Double>

  /// Initialize the random projector with a normalized projection matrix
  public init(fromShape shape: TensorShape, sampleMean: Tensor<Double>? = nil) {
    let (H, W, C) = (shape[0], shape[1], shape[2])
    
    if let mu = sampleMean {
      precondition(mu.shape == [H, W, C], "Wrong mean tensor")
      mean = mu
    } else {
      mean = Tensor(zeros: [H, W, C])
    }
  }

  /// Initialize  given an image batch
  public typealias HyperParameters = Int
  public init(from imageBatch: Tensor<Double>, given d: HyperParameters? = nil) {
    self.init(fromShape: imageBatch.shape.suffix(3), sampleMean: imageBatch.mean(squeezingAxes: 0))
  }

  /// Generate an feature from image or image batch
  /// Input: [H, W, C] or [N,H,W,C]
  /// Output: [d] or [N, d]
  @differentiable
  public func encode(_ image: Patch) -> Tensor<Double> {
    precondition(image.rank == 3 || (image.rank == 4), "wrong feature dimension \(image.shape)")
    if image.rank == 4 {
      let (N, H, W, C) = (image.shape[0], image.shape[1], image.shape[2], image.shape[3])
      let v_T = (image - mean).reshaped(to: [N, H * W * C])
      return v_T
    } else {
      let (H, W, C) = (image.shape[0], image.shape[1], image.shape[2])
      return (image - mean).reshaped(to: [H * W * C])
    }
  }
}

extension IdentityProjection: AppearanceModelEncoder {}
