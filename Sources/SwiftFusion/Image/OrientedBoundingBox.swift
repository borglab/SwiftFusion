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
  public var center: Pose2

  /// The lengths of the two sides of the region.
  public var size: Vector2

  /// Creates a instance with the given `center` and `size`.
  public init(center: Pose2, size: Vector2) {
    self.center = center
    self.size = size
  }

  /// The four corners of the region.
  public var corners: [Vector2] {
    func corner(_ xFlip: Double, _ yFlip: Double) -> Vector2 {
      return center.t + center.rot * (0.5 * Vector2(xFlip * size.x, yFlip * size.y))
    }
    return [corner(1, 1), corner(-1, 1), corner(-1, -1), corner(1, -1)]
  }
}

extension OrientedBoundingBox: Equatable {}
