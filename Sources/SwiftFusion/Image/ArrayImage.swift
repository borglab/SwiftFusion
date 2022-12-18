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
// import TensorFlow

/// Image stored as an `Array`.
///
/// Useful when doing lots of subscripting because `Array` subscripts are faster than `Tensor`
/// subscripts.
public struct ArrayImage: Differentiable {
  var pixels: [Float]
  @noDerivative let rows: Int
  @noDerivative let cols: Int
  @noDerivative let channels: Int

  /// Creates an instance of all zero values with the given `rows`, `cols`, and `channels`.
  public init(rows: Int, cols: Int, channels: Int) {
    self.pixels = Array(repeating: 0, count: rows * cols * channels)
    self.rows = rows
    self.cols = cols
    self.channels = channels
  }

  /// Creates an instance of all zero values with the same shape as `template`.
  public init(zerosLike template: Self) {
    self.pixels = Array(repeating: 0, count: template.pixels.count)
    self.rows = template.rows
    self.cols = template.cols
    self.channels = template.channels
  }

  /// Creates an instance with the given `pixels`, `rows`, `cols`, and `channels`.
  public init(pixels: [Float], rows: Int, cols: Int, channels: Int) {
    self.pixels = pixels
    self.rows = rows
    self.cols = cols
    self.channels = channels
  }

  /// Creates an instance from the given image `tensor`.
  @differentiable(reverse)
  public init(_ tensor: Tensor<Float>) {
    precondition(
      tensor.shape.count == 2 || tensor.shape.count == 3,
      "image must have shape height x width x channelCount"
    )

    self.pixels = tensor.scalars
    self.rows = tensor.shape[0]
    self.cols = tensor.shape[1]
    self.channels = tensor.shape.count == 2 ? 1 : tensor.shape[2]
  }

  /// Returns this image as an image `Tensor`.
  @differentiable(reverse)
  public var tensor: Tensor<Float> {
    Tensor(shape: [rows, cols, channels], scalars: pixels)
  }

  /// The index of a pixel in the `pixels` storage.
  ///
  /// Returns `nil` when the indices are out of bounds.
  func index(_ i: Int, _ j: Int, _ channel: Int) -> Int? {
    guard i >= 0, i < rows, j >= 0, j < cols, channel >= 0, channel < channels else {
      return nil
    }
    return i * cols * channels + j * channels + channel
  }

  /// Accesses the pixel value at `(i, j, channel)`.
  public subscript(_ i: Int, _ j: Int, _ channel: Int) -> Float {
    get {
      guard let idx = index(i, j, channel) else {
        return 0
      }
      return pixels[idx]
    }
    _modify {
      let idx = index(i, j, channel)
      assert(idx != nil, "can only modify pixels that are in bounds")
      yield &pixels[idx.unsafelyUnwrapped]
    }
  }

  /// Updates the pixel value at `(i, j, channel)` to `value`.
  ///
  /// Use this instead of the subscript when you need to differentiably modify the image.
  @differentiable(reverse)
  public mutating func update(_ i: Int, _ j: Int, _ channel: Int, to value: Float) {
    self[i, j, channel] = value
  }

  @derivative(of: update)
  @usableFromInline
  mutating func vjpUpdate(_ i: Int, _ j: Int, _ channel: Int, to value: Float) -> (
    value: (),
    pullback: (inout TangentVector) -> Float
  ) {
    update(i, j, channel, to: value)
    let idx = index(i, j, channel).unsafelyUnwrapped
    func pullback(_ v: inout TangentVector) -> Float {
      return v.pixels.base[idx]
    }
    return ((), pullback)
  }
}
