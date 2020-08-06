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
}

extension Tensor where Scalar == Double {
  /// Returns the patch of `self` at `region`.
  ///
  /// - Parameters:
  ///   - self: a pixel tensor with shape `[height, width]` or `[height, width, channelCount]`.
  ///     `height` and `width` are as defined in `docs/ImageOperations.md`.
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
}

/// Returns the pixel at `point` in `image`, using bilinear interpolation when `point` does not
/// fall exactly in the center of a pixel.
///
/// - Parameters:
///   - image: a pixel tensor, with shape `[height, width, channelCount]`. `height` and
///     `width` are as defined in `docs/ImageOperations.md`.
///   - point: a point in `(u, v)` coordinates as defined in `docs/ImageOperations.md`.
///   - channel: The channel to return.
@differentiable(wrt: point)
public func bilinear(_ image: ArrayImage, _ point: Vector2, _ channel: Int) -> Double {
  // The `(i, j)` integer coordinates of the top left pixel to sample from.
  let sourceI = withoutDerivative(at: Int(floor(point.y - 0.5)))
  let sourceJ = withoutDerivative(at: Int(floor(point.x - 0.5)))

  // Weight of the `(sourceI, _)` pixels in the interpolation.
  let weightI = Double(sourceI) + 1.5 - point.y

  // Weight of the `(_, sourceJ)` pixels in the interpolation.
  let weightJ = Double(sourceJ) + 1.5 - point.x

  let s1 = image[sourceI, sourceJ, channel] * weightI * weightJ
  let s2 = image[sourceI + 1, sourceJ, channel] * (1 - weightI) * weightJ
  let s3 = image[sourceI, sourceJ + 1, channel] * weightI * (1 - weightJ)
  let s4 = image[sourceI + 1, sourceJ + 1, channel] * (1 - weightI) * (1 - weightJ)
  return s1 + s2 + s3 + s4
}
