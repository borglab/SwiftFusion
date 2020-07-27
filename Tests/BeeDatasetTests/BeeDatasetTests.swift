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

import BeeDataset
import SwiftFusion
import TensorFlow
import XCTest

final class BeeDatasetTests: XCTestCase {
  /// Tests that we can load the frames from a fake dataset.
  func testLoadFrames() {
    let frames = BeeFrames(
      directory: datasetDirectory.appendingPathComponent("frames").appendingPathComponent("seq1"))!
    XCTAssertEqual(frames.count, 2)
    let t1 = frames[0].tensor
    XCTAssertEqual(t1.shape, [200, 100, 3])
    XCTAssertEqual(t1[0, 0], Tensor([255, 0, 0]))
    let t2 = frames[1].tensor
    XCTAssertEqual(t2.shape, [200, 100, 3])
    XCTAssertEqual(t2[0, 0], Tensor([0, 255, 0]))
  }

  /// Tests that we can load the oriented bounding boxes from a fake dataset.
  func testLoadOrientedBoundingBoxes() {
    let obbs = beeOrientedBoundingBoxes(
      file: datasetDirectory.appendingPathComponent("obbs").appendingPathComponent("seq1.txt"))!
    XCTAssertEqual(obbs.count, 2)
    XCTAssertEqual(obbs[0], OrientedBoundingBox(
      center: Pose2(Rot2(1), Vector2(100, 200)), size: Vector2(62, 28)))
    XCTAssertEqual(obbs[1], OrientedBoundingBox(
      center: Pose2(Rot2(1.5), Vector2(105, 201)), size: Vector2(62, 28)))
  }

  /// Directory of a fake dataset for tests.
  let datasetDirectory = URL.sourceFileDirectory().appendingPathComponent("fakeDataset")
}

extension URL {
  /// Creates a URL for the directory containing the caller's source file.
  static func sourceFileDirectory(file: String = #file) -> URL {
    return URL(fileURLWithPath: file).deletingLastPathComponent()
  }
}
