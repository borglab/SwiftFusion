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
  /// Returns the patch of `self` at `region`, interpolating pixels using `interpolation`.
  ///
  /// Coordinate system used in `region`: The pixel at `self[j, i, ...]` is the square bounded by
  /// the corners with coordinates `(i, j)` and `(i + 1, j + 1)`. The center of the pixel is at
  /// coorindate `(i + 0.5, j + 0.5)`.
  ///
  /// For example, `(0, 0)` is the top left of the image and `(width, height)` is the bottom right
  /// of the image.
  ///
  /// - Parameters:
  ///   - self: an image, with shape `[height, width, channelCount]`.
  ///   - region: the center, orientation, and size of the patch to extract. The center is
  ///     specified in the coordinate system described above, and the size is specified in units
  ///     of pixels.
  @differentiable
  public func patch(
    at region: OrientedBoundingBox,
    interpolation: @differentiable (Tensor<Scalar>, Scalar, Scalar) -> Tensor<Scalar>
  ) -> Tensor<Scalar> {
    precondition(self.shape.count == 3, "image must have shape height x width x channelCount")
    let patchShape: TensorShape =
      withoutDerivative(at: [Int(region.size.y), Int(region.size.x), self.shape[2]])
    var patch = Tensor<Scalar>(zeros: patchShape)
    for i in 0..<patchShape[0] {
      for j in 0..<patchShape[1] {
        let vDest = Vector2(Double(j) + 0.5, Double(i) + 0.5) - 0.5 * region.size
        let vSrc = region.center.t + region.center.rot * vDest
        patch.differentiableUpdate(i, j, to: interpolation(self, vSrc.x, vSrc.y))
      }
    }
    return patch
  }
}

/// Returns the pixel at `point` in `image`, using bilinear interpolation when `point` does not
/// fall exactly in the center of a pixel.
///
/// The coordinate system is the same as documented in `Tensor`'s `patch` method above.
///
/// Pixels that are out of bounds are considered to have the value `0` in all channels.
///
/// - Parameters:
///   - image: an image, with shape `[height, width, channelCount]`.
@differentiable
public func bilinear(_ image: Tensor<Double>, _ point: Vector2) -> Tensor<Double> {
  precondition(image.shape.count == 3)
  let i = withoutDerivative(at: Int(floor(point.y - 0.5)))
  let j = withoutDerivative(at: Int(floor(point.x - 0.5)))
  let p = Double(i) + 1.5 - point.y
  let q = Double(j) + 1.5 - point.x

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

  let s1 = pixelOrZero(image, i, j) * p * q
  let s2 = pixelOrZero(image, i + 1, j) * (1 - p) * q
  let s3 = pixelOrZero(image, i, j + 1) * p * (1 - q)
  let s4 = pixelOrZero(image, i + 1, j + 1) * (1 - p) * (1 - q)
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
