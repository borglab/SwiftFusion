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
import Foundation
import PenguinParallelWithFoundation
import XCTest

final class OISTDatasetTests: XCTestCase {
  /// Directory of a fake dataset for tests.
  let datasetDirectory = URL.sourceFileDirectory().appendingPathComponent("fakeDataset")

  /// Test that eager dataset loading works properly.
  func testEagerDatasetLoad() throws {
    if let _ = ProcessInfo.processInfo.environment["CI"] {
      throw XCTSkip("Test skipped on CI because it downloads a lot of data.")
    }
    ComputeThreadPools.local =
      NonBlockingThreadPool<PosixConcurrencyPlatform>(name: "mypool", threadCount: 5)

    // Truncate the frames so that this test does not take a huge amount of time.
    let video = OISTBeeVideo(directory: datasetDirectory, length: 2)!

    XCTAssertEqual(video.frames.count, 2)
    XCTAssertNotEqual(video.frames[0], video.frames[1])

    // There are fewer tracks because we truncated the frames.
    XCTAssertEqual(video.tracks.count, 1)

    // The tracks are shorter because we truncated the frames.
    XCTAssertEqual(video.tracks[0].boxes.count, 2)
  }

  func testToString() {
    let label = OISTBeeLabel(
      frameIndex: 1515,
      label: .Body,
      rawLocation: (1.2, 38, 1.13559),
      offset: (114514, 1919810)
    )
    
    let string = label.toString()
    let expected = "114514\t1919810\t1\t1.2\t38.0\t1.13559"
    XCTAssertEqual(string, expected)
  }
}
