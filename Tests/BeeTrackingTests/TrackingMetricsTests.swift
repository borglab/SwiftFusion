
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
import PythonKit
import XCTest

import BeeTracking
import SwiftFusion

class TrackingMetricsTestsests: XCTestCase {
  /// Test tracking metrics for a subsequence that tracks ground truth perfectly.
  func testPerfectSubsequence() throws {
    guard let _ = try? Python.attemptImport("shapely") else {
      throw XCTSkip("overlap requires shapely python library")
    }

    let groundTruth = Array(repeating: box(Pose2()), count: 10)
    let metrics = SubsequenceMetrics(groundTruth: groundTruth, prediction: groundTruth)

    XCTAssertEqual(metrics.accuracy, 1)
    XCTAssertEqual(metrics.robustness, 1)
    XCTAssertEqual(metrics.NFsa, 10)
    XCTAssertEqual(metrics.Nsa, 10)
    for i in 0..<10 {
      XCTAssertEqual(metrics.extendedAverageOverlap(i), 1)
    }
    XCTAssertEqual(metrics.extendedAverageOverlap(10), nil)
  }

  /// Test tracking metrics for a subsequence.
  func testSubsequence() {
    let groundTruth = Array(repeating: box(Pose2()), count: 7)
    let prediction = [
      box(Pose2()), box(Pose2(1, 0, 0)), box(Pose2(2, 0, 0)), box(Pose2(3, 0, 0)),
      box(Pose2(100, 0, 0)), box(Pose2(100, 0, 0)), box(Pose2())
    ]
    let metrics = SubsequenceMetrics(groundTruth: groundTruth, prediction: prediction)

    let overlaps = zip(groundTruth, prediction).map { $0.0.overlap($0.1) }

    XCTAssertEqual(metrics.accuracy, overlaps[0..<4].reduce(0, +) / 4)
    XCTAssertEqual(metrics.robustness, 4 / 7)
    XCTAssertEqual(metrics.NFsa, 4)
    XCTAssertEqual(metrics.Nsa, 7)
    for i in 0..<4 {
      XCTAssertEqual(
        metrics.extendedAverageOverlap(i),
        overlaps[0..<(i+1)].reduce(0, +) / Double(i + 1))
    }
    for i in 4..<10 {
      XCTAssertEqual(metrics.extendedAverageOverlap(i), 0)
    }
  }

  /// Test tracking metrics for a sequence with multiple subsequences.
  func testSequence() {
    let sub1 = SubsequenceMetrics(
      accuracy: 1, robustness: 1, NFsa: 10, Nsa: 10,
      averageOverlap: Array(repeating: 1, count: 10))
    let sub2 = SubsequenceMetrics(
      accuracy: 0.4, robustness: 1, NFsa: 5, Nsa: 5,
      averageOverlap: Array(repeating: 0.4, count: 5))
    let sub3 = SubsequenceMetrics(
      accuracy: 1, robustness: 0.5, NFsa: 4, Nsa: 8,
      averageOverlap: Array(repeating: 1, count: 4))
    let metrics = SequenceMetrics([sub1, sub2, sub3])

    XCTAssertEqual(metrics.accuracy, 16 / 19)
    XCTAssertEqual(metrics.robustness, 19 / 23)
    XCTAssertEqual(metrics.NFs, 19)
    XCTAssertEqual(metrics.Ns, 10)
  }

  /// Test EAO metric for a collection of subsequences.
  func testEAO() {
    let sub1 = SubsequenceMetrics(
      accuracy: 1, robustness: 1, NFsa: 10, Nsa: 10,
      averageOverlap: Array(repeating: 1, count: 10))
    let sub2 = SubsequenceMetrics(
      accuracy: 0.4, robustness: 1, NFsa: 5, Nsa: 5,
      averageOverlap: Array(repeating: 0.4, count: 5))
    let sub3 = SubsequenceMetrics(
      accuracy: 1, robustness: 0.5, NFsa: 4, Nsa: 8,
      averageOverlap: Array(repeating: 1, count: 4))
    let eao = ExpectedAverageOverlap([sub1, sub2, sub3])

    XCTAssertEqual(eao.curve.count, 10)
    XCTAssertEqual(eao.curve[0], 2.4 / 3)
    XCTAssertEqual(eao.curve[1], 2.4 / 3)
    XCTAssertEqual(eao.curve[2], 2.4 / 3)
    XCTAssertEqual(eao.curve[3], 2.4 / 3)
    XCTAssertEqual(eao.curve[4], 1.4 / 3)
    XCTAssertEqual(eao.curve[5], 1 / 2)
    XCTAssertEqual(eao.curve[6], 1 / 2)
    XCTAssertEqual(eao.curve[7], 1 / 2)
    XCTAssertEqual(eao.curve[8], 1 / 2)
    XCTAssertEqual(eao.curve[9], 1 / 2)
  }
}

fileprivate func box(_ pose: Pose2) -> OrientedBoundingBox {
  return OrientedBoundingBox(center: pose, rows: 10, cols: 10)
}
