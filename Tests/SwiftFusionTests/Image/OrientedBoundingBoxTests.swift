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

import SwiftFusion
import XCTest

final class OrientedBoundingBoxTests: XCTestCase {
  /// Test that we calculate the correct corners for the bounding box.
  func testCorners() {
    let box1 = OrientedBoundingBox(
      center: Pose2(Rot2(0), Vector2(5, 10)), size: Vector2(100, 200))
    XCTAssertEqual(box1.corners, [
      Vector2(55, 110),
      Vector2(-45, 110),
      Vector2(-45, -90),
      Vector2(55, -90)
    ])

    let box2 = OrientedBoundingBox(
      center: Pose2(Rot2(.pi / 2), Vector2(5, 10)), size: Vector2(100, 200))
    assertAllKeyPathEqual(box2.corners, [
      Vector2(-95, 60),
      Vector2(-95, -40),
      Vector2(105, -40),
      Vector2(105, 60)
    ], accuracy: 1e-5)
  }
}
