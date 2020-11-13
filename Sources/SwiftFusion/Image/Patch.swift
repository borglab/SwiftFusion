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
  ///   - outputSize: The size, in `(height, width)` pixels, to scale the result to. If this is not
  ///                 specified, then the result has the same size as `region`.
  @differentiable(wrt: region)
  public func patch(at region: OrientedBoundingBox, outputSize: (Int, Int)? = nil) -> Self {
    let (outputRows, outputCols) = outputSize ?? (region.rows, region.cols)
    let rowScale = Double(region.rows) / Double(outputRows)
    let colScale = Double(region.cols) / Double(outputCols)
    var result = ArrayImage(rows: outputRows, cols: outputCols, channels: self.channels)
    for i in 0..<outputRows {
      for j in 0..<outputCols {
        // The position of the destination pixel in the source region, in `(u, v)` coordinates.
        let uvSrcRegion = Vector2(colScale * (Double(j) + 0.5), rowScale * (Double(i) + 0.5))

        // The position of the destination pixel in the destination image, in coordinates where the
        // center of the destination image is `(0, 0)`.
        let xySrcRegion = uvSrcRegion - 0.5 * Vector2(Double(region.cols), Double(region.rows))

        for c in 0..<self.channels {
          result.update(i, j, c, to: bilinear(self, region.center * xySrcRegion, c))
        }
      }
    }
    return result
  }

  @derivative(of: patch, wrt: region)
  @usableFromInline
  func vjpPatch(
    at region: OrientedBoundingBox, outputSize: (Int, Int)? = nil
  ) -> (value: Self, pullback: (TangentVector) -> OrientedBoundingBox.TangentVector) {
    let r = self.patch(at: region, outputSize: outputSize)
    let channels = self.channels
    let (outputRows, outputCols) = outputSize ?? (Int(region.rows), Int(region.cols))
    let rowScale = Double(region.rows) / Double(outputRows)
    let colScale = Double(region.cols) / Double(outputCols)
    func outerPb(_ dOut: TangentVector) -> OrientedBoundingBox.TangentVector {
      var idx = 0
      var dBox = region.zeroTangentVector
      for i in 0..<outputRows {
        for j in 0..<outputCols {
          // The position of the destination pixel in the source region, in `(u, v)`
          // coordinates.
          let uvSrcRegion =
            Vector2(colScale * (Double(j) + 0.5), rowScale * (Double(i) + 0.5))

          // The position of the destination pixel in the destination image, in coordinates
          // where the center of the destination image is `(0, 0)`.
          let xySrcRegion = uvSrcRegion - 0.5 * Vector2(Double(region.cols), Double(region.rows))

          for c in 0..<channels {
            if dOut.pixels.base[idx] != 0 {
              let pb = pullback(at: region) {
                bilinear(self, $0.center * xySrcRegion, c)
              }
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
  ///   - outputSize: The size, in `(height, width)` pixels, to scale the result to. If this is not
  ///                 specified, then the result has the same size as `region`.
  ///
  /// - Returns:
  ///   - patch: The region at `region`, cropped from `self`.
  ///   - jacobian: The directional derivatives `dtheta`, `du`, and `dv` along the rotation and
  ///               translation directions respectively. These can be thought of as the columns of
  ///               the Jacobian.
  public func patchWithJacobian(at region: OrientedBoundingBox, outputSize: (Int, Int)? = nil)
    -> (patch: Self, jacobian: (dtheta: Self, du: Self, dv: Self))
  {
    let (outputRows, outputCols) = outputSize ?? (region.rows, region.cols)
    let rowScale = Double(region.rows) / Double(outputRows)
    let colScale = Double(region.cols) / Double(outputCols)
    var result = ArrayImage(rows: outputRows, cols: outputCols, channels: self.channels)
    var dtheta = ArrayImage(zerosLike: result)
    var du = ArrayImage(zerosLike: result)
    var dv = ArrayImage(zerosLike: result)
    for i in 0..<outputRows {
      for j in 0..<outputCols {
        // The position of the destination pixel in the source region, in `(u, v)` coordinates.
        let uvSrcRegion = Vector2(colScale * (Double(j) + 0.5), rowScale * (Double(i) + 0.5))

        // The position of the destination pixel in the destination image, in coordinates where the
        // center of the destination image is `(0, 0)`.
        let xySrcRegion = uvSrcRegion - 0.5 * Vector2(Double(region.cols), Double(region.rows))

        // The position of the destination pixel in the source image. These are the `(u, v)`
        // coordinates of the source image that we sample from.
        let uvSrc: Vector2 = region.center * xySrcRegion

        // The derivatives of `uvSrc` with respect to the rotation and translation components of
        // `region.center`.
        //
        // This follows from equation (11.2) in section "11 2D Rigid Transformations" of [1]:
        //   Substitute
        //     (R, t) = (region.center.rot, region.center.t)
        //     p = xySrcRegion
        //     q = uvSrc
        //   Then (11.2) says that the Jacobian of the action of `region.center` on `xyDest` is
        //     [ R  R*R_pi/2*p
        //       0           0 ]
        //   The first two columns of the Jacobian give us the derivatives with respect to the
        //   translation components of `region.center` and the last column gives us the derivative
        //   with respect to the rotation component of `region.center`.
        //
        // [1] https://github.com/borglab/gtsam/blob/develop/doc/math.pdf
        let duvSrc_dtheta = region.center.rot * Vector2(-xySrcRegion.y, xySrcRegion.x)
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

extension Tensor where Scalar: TensorFlowFloatingPoint {
  /// Returns the patch of `self` at `region`.
  ///
  /// - Parameters:
  ///   - self: a tensor of image pixels with shape `[height, width]` or
  ///           [height, width, channelCount]`. `height` and `width` are as defined in
  ///           `docs/ImageOperations.md`.
  ///   - region: the center, orientation, and size of the patch to extract.
  ///   - outputSize: The size, in `(height, width)` pixels, to scale the result to. If this is not
  ///                 specified, then the result has the same size as `region`.
  @differentiable(wrt: region)
  public func patch(at region: OrientedBoundingBox, outputSize: (Int, Int)? = nil)
    -> Tensor<Scalar>
  {
    precondition(
      self.shape.count == 2 || self.shape.count == 3,
      "image must have shape [height, width] or [height, width, channelCount]"
    )
    let result = Tensor<Scalar>(
      ArrayImage(Tensor<Float>(self)).patch(at: region, outputSize: outputSize).tensor)
    return self.shape.count == 2 ? result.reshaped(to: [result.shape[0], result.shape[1]]) : result
  }

  /// Returns the patch of `self` at `region`, and its Jacobian with respect to the pose of the
  /// `region`.
  ///
  /// - Parameters:
  ///   - self: a tensor of image pixels with shape `[height, width, channelCount]`. `height`
  ///           and `width` are as defined in `docs/ImageOperations.md`.
  ///   - region: the center, orientation, and size of the patch to extract.
  ///   - outputSize: The size, in `(height, width)` pixels, to scale the result to. If this is not
  ///                 specified, then the result has the same size as `region`.
  ///
  /// - Returns:
  ///   - patch: The region at `region`, cropped from `self`.
  ///   - jacobian: The derivative of the patch operation with respect to `region`. Has shape
  ///               `[height, width, channelCount, 3]`. The slices along the last dimension are
  ///                the directional derivatives along the rotation and the `u`, `v` translation
  ///                components (in that order).
  public func patchWithJacobian(at region: OrientedBoundingBox, outputSize: (Int, Int)? = nil)
    -> (patch: Tensor<Scalar>, jacobian: Tensor<Scalar>)
  {
    precondition(self.shape.count == 3, "image must have shape [height, width, channelCount]")
    let (patch, jacobian) = ArrayImage(Tensor<Float>(self)).patchWithJacobian(at: region, outputSize: outputSize)
    return (
      patch: Tensor<Scalar>(patch.tensor),
      jacobian: Tensor<Scalar>(Tensor<Float>(stacking: [
        jacobian.dtheta.tensor,
        jacobian.du.tensor,
        jacobian.dv.tensor
      ], alongAxis: -1)))
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
public func bilinear(_ source: ArrayImage, _ point: Vector2, _ channel: Int) -> Float {
  // The `(i, j)` integer coordinates of the top left pixel to sample from.
  let sourceI = withoutDerivative(at: Int(floor(point.y - 0.5)))
  let sourceJ = withoutDerivative(at: Int(floor(point.x - 0.5)))

  // Weight of the `(sourceI, _)` pixels in the interpolation.
  let weightI = Float(sourceI) + 1.5 - Float(differentiableConversion: point.y)

  // Weight of the `(_, sourceJ)` pixels in the interpolation.
  let weightJ = Float(sourceJ) + 1.5 - Float(differentiableConversion: point.x)

  let s1 = source[sourceI, sourceJ, channel] * weightI
  let s2 = source[sourceI + 1, sourceJ, channel] * (1 - weightI)
  let s3 = source[sourceI, sourceJ + 1, channel] * weightI
  let s4 = source[sourceI + 1, sourceJ + 1, channel] * (1 - weightI)
  return (s1 + s2) * weightJ + (s3 + s4) * (1 - weightJ)
}

/// Returns the directional derivative of `bilinear` with respect to `point`, in direction `dPoint`.
///
/// The parameters other than `dPoint` correspond to the parameters of `bilinear`.
public func dBilinear(
  _ source: ArrayImage, _ point: Vector2, _ channel: Int, _ dPoint: Vector2
) -> Float {
  // The `(i, j)` integer coordinates of the top left pixel to sample from.
  let sourceI = withoutDerivative(at: Int(floor(point.y - 0.5)))
  let sourceJ = withoutDerivative(at: Int(floor(point.x - 0.5)))

  // Weight of the `(sourceI, _)` pixels in the interpolation.
  let weightI = Float(sourceI) + 1.5 - Float(point.y)

  // Weight of the `(_, sourceJ)` pixels in the interpolation.
  let weightJ = Float(sourceJ) + 1.5 - Float(point.x)

  let dWeightI = Float(-dPoint.y)
  let dWeightJ = Float(-dPoint.x)

  let s1 = source[sourceI, sourceJ, channel] * (dWeightI * weightJ + weightI * dWeightJ)
  let s2 = source[sourceI + 1, sourceJ, channel] * (-dWeightI * weightJ + (1 - weightI) * dWeightJ)
  let s3 = source[sourceI, sourceJ + 1, channel] * (dWeightI * (1 - weightJ) + weightI * -dWeightJ)
  let s4 = source[sourceI + 1, sourceJ + 1, channel]
    * (-dWeightI * (1 - weightJ) + (1 - weightI) * -dWeightJ)
  return s1 + s2 + s3 + s4
}

fileprivate extension Float {
  @differentiable
  init(differentiableConversion x: Double) {
    self.init(x)
  }

  @derivative(of: init(differentiableConversion:))
  static func vjpInit(differentiableConversion x: Double) -> (
    value: Float, pullback: (Float) -> Double
  ) {
    (Float(x), { Double($0) })
  }
}
