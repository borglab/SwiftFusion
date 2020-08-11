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

extension ArrayImage {
  /// Returns the patch of `self` at `region`.
  ///
  /// - Parameters:
  ///   - region: the center, orientation, and size of the patch to extract.
  @differentiable(wrt: region)
  public func patch(at region: OrientedBoundingBox) -> Self {
    var result = ArrayImage(rows: region.rows, cols: region.cols, channels: self.channels)
    for i in 0..<region.rows {
      for j in 0..<region.cols {
        // The position of the destination pixel in the destination image, in `(u, v)` coordinates.
        let uvDest = Vector2(Double(j) + 0.5, Double(i) + 0.5)

        // The position of the destination pixel in the source image, in coordinates where the
        // center of the destination image is `(0, 0)`.
        let xyDest = uvDest - 0.5 * Vector2(Double(region.cols), Double(region.rows))

        for c in 0..<self.channels {
          result.update(i, j, c, to: bilinear(self, region.center * xyDest, c))
        }
      }
    }
    return result
  }

  @derivative(of: patch, wrt: region)
  @usableFromInline
  func vjpPatch(at region: OrientedBoundingBox) -> (value: Self, pullback: (TangentVector) -> OrientedBoundingBox.TangentVector) {
    let r = self.patch(at: region)
    let channels = self.channels
    func outerPb(_ dOut: TangentVector) -> OrientedBoundingBox.TangentVector {
      var idx = 0
      var dBox = OrientedBoundingBox.TangentVector.zero
      for i in 0..<region.rows {
        for j in 0..<region.cols {
          // The position of the destination pixel in the destination image, in `(u, v)` coordinates.
          let uvDest = Vector2(Double(j) + 0.5, Double(i) + 0.5)

          // The position of the destination pixel in the source image, in coordinates where the
          // center of the destination image is `(0, 0)`.
          let xyDest = uvDest - 0.5 * Vector2(Double(region.cols), Double(region.rows))

          for c in 0..<channels {
            if dOut.pixels.base[idx] != 0 {
              let pb = pullback(at: region) { bilinear(self, $0.center * xyDest, c) }
              dBox += pb(dOut.pixels.base[idx])
            }
            idx += 1
          }
        }
      }
      return dBox
    }
    return (r, outerPb)
  }

  /// Returns the patch of `self` at `region`, and its Jacobian with respect to the center of the
  /// `region`.
  ///
  /// - Parameters:
  ///   - region: the center, orientation, and size of the patch to extract.
  ///
  /// - Returns:
  ///   - patch: The region at `region`, cropped from `self`.
  ///   - jacobian: The directional derivatives `dtheta`, `du`, and `dv` along the rotation and
  ///               translation directions respectively. These can be thought of as the columns of
  ///               the Jacobian.
  public func patchWithJacobian(at region: OrientedBoundingBox)
    -> (patch: Self, jacobian: (dtheta: Self, du: Self, dv: Self))
  {
    var result = ArrayImage(rows: region.rows, cols: region.cols, channels: self.channels)
    var dtheta = ArrayImage(rows: region.rows, cols: region.cols, channels: self.channels)
    var du = ArrayImage(rows: region.rows, cols: region.cols, channels: self.channels)
    var dv = ArrayImage(rows: region.rows, cols: region.cols, channels: self.channels)
    for i in 0..<region.rows {
      for j in 0..<region.cols {
        // The position of the destination pixel in the destination image, in `(u, v)` coordinates.
        let uvDest = Vector2(Double(j) + 0.5, Double(i) + 0.5)

        // The position of the destination pixel in the source image, in coordinates where the
        // center of the destination image is `(0, 0)`.
        let xyDest = uvDest - 0.5 * Vector2(Double(region.cols), Double(region.rows))

        let uvSrc: Vector2 = region.center * xyDest

        let duvSrc_dtheta = region.center.rot * Vector2(-xyDest.y, xyDest.x)
        let duvSrc_du = region.center.rot * Vector2(1, 0)
        let duvSrc_dv = region.center.rot * Vector2(0, 1)

        for c in 0..<self.channels {
          result[i, j, c] = bilinear(self, uvSrc, c)
          dtheta[i, j, c] = dBilinear(self, uvSrc, c, duvSrc_dtheta)
          du[i, j, c] = dBilinear(self, uvSrc, c, duvSrc_du)
          dv[i, j, c] = dBilinear(self, uvSrc, c, duvSrc_dv)
        }
      }
    }
    return (result, (dtheta, du, dv))
  }
}

extension Tensor where Scalar == Double {
  /// Returns the patch of `self` at `region`.
  ///
  /// - Parameters:
  ///   - self: a tensor of image pixels with shape `[height, width]` or
  ///           [height, width, channelCount]`. `height` and `width` are as defined in
  ///           `docs/ImageOperations.md`.
  ///   - region: the center, orientation, and size of the patch to extract.
  @differentiable(wrt: region)
  public func patch(at region: OrientedBoundingBox) -> Tensor<Scalar> {
    precondition(
      self.shape.count == 2 || self.shape.count == 3,
      "image must have shape [height, width] or [height, width, channelCount]"
    )
    let result = ArrayImage(self).patch(at: region).tensor
    return self.shape.count == 2 ? result.reshaped(to: [region.rows, region.cols]) : result
  }

  /// Returns the patch of `self` at `region`, and its Jacobian with respect to the center of the
  /// `region`.
  ///
  /// - Parameters:
  ///   - self: a tensor of image pixels with shape `[height, width, channelCount]`. `height`
  ///           and `width` are as defined in `docs/ImageOperations.md`.
  ///   - region: the center, orientation, and size of the patch to extract.
  ///
  /// - Returns:
  ///   - patch: The region at `region`, cropped from `self`.
  ///   - jacobian: The derivative of the patch operation with respect to `region`. Has shape
  ///               `[height, width, channelCount, 3]`. The slices along the last dimension are
  ///                the directional derivatives along the rotation and the `u`, `v` translation
  ///                components (in that order).
  public func patchWithJacobian(at region: OrientedBoundingBox)
    -> (patch: Tensor<Scalar>, jacobian: Tensor<Scalar>)
  {
    precondition(self.shape.count == 3, "image must have shape [height, width, channelCount]")
    let (patch, jacobian) = ArrayImage(self).patchWithJacobian(at: region)
    return (
      patch: patch.tensor,
      jacobian: Tensor<Double>(stacking: [
        jacobian.dtheta.tensor,
        jacobian.du.tensor,
        jacobian.dv.tensor
      ], alongAxis: -1))
  }
}

/// Returns the `channel`-th component of the pixel at `point` in `source`, using bilinear
/// interpolation when `point` does not fall exactly in the center of a pixel.
///
/// - Parameters:
///   - source: a pixel tensor, with shape `[height, width, channelCount]`. `height` and
///     `width` are as defined in `docs/ImageOperations.md`.
///   - point: a point in `(u, v)` coordinates as defined in `docs/ImageOperations.md`.
///   - channel: The channel to return.
@differentiable(wrt: point)
public func bilinear(_ source: ArrayImage, _ point: Vector2, _ channel: Int) -> Double {
  // The `(i, j)` integer coordinates of the top left pixel to sample from.
  let sourceI = withoutDerivative(at: Int(floor(point.y - 0.5)))
  let sourceJ = withoutDerivative(at: Int(floor(point.x - 0.5)))

  // Weight of the `(sourceI, _)` pixels in the interpolation.
  let weightI = Double(sourceI) + 1.5 - point.y

  // Weight of the `(_, sourceJ)` pixels in the interpolation.
  let weightJ = Double(sourceJ) + 1.5 - point.x

  let s1 = source[sourceI, sourceJ, channel] * weightI * weightJ
  let s2 = source[sourceI + 1, sourceJ, channel] * (1 - weightI) * weightJ
  let s3 = source[sourceI, sourceJ + 1, channel] * weightI * (1 - weightJ)
  let s4 = source[sourceI + 1, sourceJ + 1, channel] * (1 - weightI) * (1 - weightJ)
  return s1 + s2 + s3 + s4
}

/// Returns the directional derivative of `bilinear` with respect to `point`, in direction `dPoint`.
///
/// The parameters other than `dPoint` correspond to the parameters of `bilinear`.
public func dBilinear(
  _ source: ArrayImage, _ point: Vector2, _ channel: Int, _ dPoint: Vector2
) -> Double {
  // The `(i, j)` integer coordinates of the top left pixel to sample from.
  let sourceI = withoutDerivative(at: Int(floor(point.y - 0.5)))
  let sourceJ = withoutDerivative(at: Int(floor(point.x - 0.5)))

  // Weight of the `(sourceI, _)` pixels in the interpolation.
  let weightI = Double(sourceI) + 1.5 - point.y

  // Weight of the `(_, sourceJ)` pixels in the interpolation.
  let weightJ = Double(sourceJ) + 1.5 - point.x

  let dWeightI = -dPoint.y
  let dWeightJ = -dPoint.x

  let s1 = source[sourceI, sourceJ, channel] * (dWeightI * weightJ + weightI * dWeightJ)
  let s2 = source[sourceI + 1, sourceJ, channel] * (-dWeightI * weightJ + (1 - weightI) * dWeightJ)
  let s3 = source[sourceI, sourceJ + 1, channel] * (dWeightI * (1 - weightJ) + weightI * -dWeightJ)
  let s4 = source[sourceI + 1, sourceJ + 1, channel]
    * (-dWeightI * (1 - weightJ) + (1 - weightI) * -dWeightJ)
  return s1 + s2 + s3 + s4
}
