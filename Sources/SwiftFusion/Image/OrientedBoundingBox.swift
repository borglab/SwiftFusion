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

/// A rectangular region of an image, not necessarily axis-aligned.
public struct OrientedBoundingBox: Differentiable {
  /// The pose of the region's center within the image.
  ///
  /// The translation component is in `(u, v)` coordinates as defined in `docs/ImageOperations.md`.
  @differentiable
  public var center: Pose2

  /// The number of pixels along the height axis.
  ///
  /// This is the `rows` image dimension defined in `docs/ImageOperations.md`.
  @noDerivative public let rows: Int

  /// The number of pixels along the width axis.
  ///
  /// This is the `cols` image dimension defines in `docs/ImageOperations.md`.
  @noDerivative public let cols: Int

  /// Creates a instance with the given `center`, `rows`, and `cols`.
  @differentiable
  public init(center: Pose2, rows: Int, cols: Int) {
    self.center = center
    self.rows = rows
    self.cols = cols
  }

  /// The four corners of the region, in `(u, v)` coordinates as defined in
  /// `docs/ImageOperations.md`.
  @differentiable
  public var corners: [Vector2] {
    /// Returns a corner of `self`.
    ///
    /// - Parameter `uFlip`: `-1` or `1`, determines which side of `self`'s `u`-axis the corner is on.
    /// - Parameter `vFlip`: `-1` or `1`, determines which side of `self`'s `v`-axis the corner is on.
    func corner(_ uFlip: Double, _ vFlip: Double) -> Vector2 {
      return center.t + center.rot * (0.5 * Vector2(uFlip * Double(cols), vFlip * Double(rows)))
    }
    return [corner(1, 1), corner(-1, 1), corner(-1, -1), corner(1, -1)]
  }
}

extension OrientedBoundingBox: Equatable {}
