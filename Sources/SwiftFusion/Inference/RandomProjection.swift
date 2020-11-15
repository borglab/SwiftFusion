
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

/// A class performing common activities in the RandomProjection framework.
/// Input shape for training: [N, H, W, C]
/// W matrix: [feature, H*W*C]
/// Output: [feature]
public struct RandomProjection {
  public typealias Patch = Tensor<Double>

  /// Random Basis Matrix
  /// When input image is of shape [N, H, W, C]
  /// B is of shape [d, H * W * C]
  public let B: Tensor<Double>

  /// Initialize the random projector with a normalized projection matrix
  public init(fromShape shape: TensorShape, toFeatureSize d: Int) {
    let (H, W, C) = (shape[0], shape[1], shape[2])
    B = Tensor<Double>(
      stacking: (0..<d).map { _ in
        let t = Tensor<Double>(randomNormal: [H * W * C])
        return t/sqrt(t.squared().sum())
      }
    )
  }

  /// Generate an feature from image or image batch
  /// Input: [H, W, C] or [N,H,W,C]
  /// Output: [d] or [N, d]
  @differentiable
  public func encode(_ image: Patch) -> Tensor<Double> {
    precondition(image.rank == 3 || (image.rank == 4), "wrong feature dimension \(image.shape)")
    let HWC = B.shape[1]
    let d = B.shape[0]
    if image.rank == 4 {
      let N = image.shape[0]
      let v_T = (image).reshaped(to: [HWC, N]).transposed()
      return matmul(v_T, B.transposed()).reshaped(to: [N, d])
    } else {
      return matmul(B, (image).reshaped(to: [HWC, 1])).reshaped(to: [d])
    }
  }
}

extension RandomProjection: AppearanceModelEncoder {}
