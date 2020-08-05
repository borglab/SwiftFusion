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

extension Tensor where Scalar == Double {
  /// Returns the patch of `self` at `region`, sampling pixels using `resample`.
  ///
  /// - Parameters:
  ///   - self: a pixel tensor with shape `[height, width, channelCount]`. `height` and `width` are
  ///     as defined in `docs/ImageOperations.md`.
  ///   - region: the center, orientation, and size of the patch to extract.
  ///   - resample: Returns a sample of `image` at `point`.
  ///     - image: a pixel tensor, with shape `[height, width, channelCount]`. `height` and
  ///       `width` are as defined in `docs/ImageOperations.md`.
  ///     - point: a point in `(u, v)` coordinates as defined in `docs/ImageOperations.md`.
  @differentiable
  public func patch(
    at region: OrientedBoundingBox,
    resample: @differentiable (_ image: Tensor<Scalar>, _ point: Vector2) -> Tensor<Scalar>
      = bilinear
  ) -> Tensor<Scalar> {
    precondition(self.shape.count == 3, "image must have shape height x width x channelCount")
    let patchShape: TensorShape = [region.rows, region.cols, self.shape[2]]
    var patch = Tensor<Scalar>(zeros: patchShape)
    for i in 0..<region.rows {
      for j in 0..<region.cols {
        // The position of the destination pixel in the destination image, in `(u, v)` coordinates.
        let uvDest = Vector2(Double(j) + 0.5, Double(i) + 0.5)

        // The position of the destination pixel in the destination image, in coordinates where the
        // center of the destination image is `(0, 0)`.
        let xyDest = uvDest - 0.5 * Vector2(Double(region.cols), Double(region.rows))

        patch.differentiableUpdate(i, j, to: resample(self, region.center * xyDest))
      }
    }
    return patch
  }
}

/// Returns the pixel at `point` in `image`, using bilinear interpolation when `point` does not
/// fall exactly in the center of a pixel.
///
/// - Parameters:
///   - image: a pixel tensor, with shape `[height, width, channelCount]`. `height` and
///     `width` are as defined in `docs/ImageOperations.md`.
///   - point: a point in `(u, v)` coordinates as defined in `docs/ImageOperations.md`.
@differentiable
public func bilinear(_ image: Tensor<Double>, _ point: Vector2) -> Tensor<Double> {
  precondition(image.shape.count == 3)

  // The `(i, j)` integer coordinates of the top left pixel to sample from.
  let i = withoutDerivative(at: Int(floor(point.y - 0.5)))
  let j = withoutDerivative(at: Int(floor(point.x - 0.5)))

  // Weight of the `(i, _)` pixels in the interpolation.
  let weightI = Double(i) + 1.5 - point.y

  // Weight of the `(_, j)` pixels in the interpolation.
  let weightJ = Double(j) + 1.5 - point.x

  func pixelOrZero(_ t: Tensor<Double>, _ i: Int, _ j: Int) -> Tensor<Double> {
    if i < 0 {
      return Tensor(zeros: [t.shape[2]])
    }
    if j < 0 {
      return Tensor(zeros: [t.shape[2]])
    }
    if i >= t.shape[0] {
      return Tensor(zeros: [t.shape[2]])
    }
    if j >= t.shape[1] {
      return Tensor(zeros: [t.shape[2]])
    }
    return t[i, j]
  }

  let s1 = pixelOrZero(image, i, j) * weightI * weightJ
  let s2 = pixelOrZero(image, i + 1, j) * (1 - weightI) * weightJ
  let s3 = pixelOrZero(image, i, j + 1) * weightI * (1 - weightJ)
  let s4 = pixelOrZero(image, i + 1, j + 1) * (1 - weightI) * (1 - weightJ)
  return s1 + s2 + s3 + s4
}

/// Implements a differentiable update method because the normal subscript setter does not have a
/// derivative.
fileprivate extension Tensor where Scalar: TensorFlowFloatingPoint {
  @differentiable
  mutating func differentiableUpdate(_ i: Int, _ j: Int, to value: Self) {
    precondition(self.shape.count == 3)
    self[i, j] = value
  }

  @derivative(of: differentiableUpdate)
  mutating func vjpDifferentiableUpdate(_ i: Int, _ j: Int, to value: Self)
    -> (value: (), pullback: (inout Self) -> Self)
  {
    self.differentiableUpdate(i, j, to: value)
    let valueShape = value.shape
    func pullback(_ t: inout Self) -> Self {
      let tValue = t[i, j]
      t[i, j] = Tensor(zeros: valueShape)
      return tValue
    }
    return ((), pullback)
  }
}
