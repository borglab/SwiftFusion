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

import PenguinParallelWithFoundation
import TensorFlow
import XCTest

import BeeDataset
import BeeTracking

class OISTBeeVideoBatchesTests: XCTestCase {
  /// Tests getting a batch of bee patches.
  func testBeeBatch() {
   ComputeThreadPools.local =
        NonBlockingThreadPool<PosixConcurrencyPlatform>(name: "mypool", threadCount: 5)
    let video = OISTBeeVideo(deferLoadingFrames: true)!
    let (batch, statistics) = video.makeBatch(appearanceModelSize: (100, 100), batchSize: 200)
    XCTAssertEqual(batch.shape, [200, 100, 100, 3])
    XCTAssertEqual(statistics.mean.shape, [])
    XCTAssertEqual(statistics.standardDeviation.shape, [])
  }

  /// Tests getting a batch of background patches.
  func testBackgroundBatch() {
    ComputeThreadPools.local =
      NonBlockingThreadPool<PosixConcurrencyPlatform>(name: "mypool", threadCount: 5)
    let video = OISTBeeVideo(deferLoadingFrames: true)!
    let (batch, statistics) = video.makeBackgroundBatch(
      patchSize: (40, 70), appearanceModelSize: (100, 100), batchSize: 200)
    XCTAssertEqual(batch.shape, [200, 100, 100, 3])
    XCTAssertEqual(statistics.mean.shape, [])
    XCTAssertEqual(statistics.standardDeviation.shape, [])
  }
}
