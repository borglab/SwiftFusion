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

final class G2OReaderTests: XCTestCase {
  /// Tests loading a simple g2o file using G2OReader.
  func testLoadSimple() {
    var dataset = G2OArray()
    try! dataset.read(
      fromG2O: URL(fileURLWithPath: "Tests/SwiftFusionTests/Dataset/Data/simple.g2o"))
    let expectedDataset = G2OArray(
      initialGuesses: [
        G2OArray.InitialGuess(index: 0, pose: Pose2(0.1, 0.2, 0.3)),
        G2OArray.InitialGuess(index: 1, pose: Pose2(0.4, 0.5, 0.6)),
      ],
      measurements: [
        G2OArray.Measurement(frameIndex: 0, measuredIndex: 1, pose: Pose2(0.7, 0.8, 0.9))
      ]
    )
    XCTAssertEqual(dataset, expectedDataset)
  }

  /// Tests that an error is thrown when reading a file that does not exist.
  func testLoadNotExist() {
    var dataset = G2OArray()
    XCTAssertThrowsError(
      try dataset.read(
        fromG2O: URL(fileURLWithPath: "Tests/SwiftFusionTests/Dataset/Data/notexist.g2o"))
    )
  }

  /// Tests that an error is thrown when reading a malformed g2o file.
  func testLoadMalformed() {
    var dataset = G2OArray()
    XCTAssertThrowsError(
      try dataset.read(
        fromG2O: URL(fileURLWithPath: "Tests/SwiftFusionTests/Dataset/Data/malformed.g2o"))
    )
  }

  static var allTests = [
    ("testLoadSimple", testLoadSimple),
    ("testLoadNotExist", testLoadNotExist),
    ("testLoadMalformed", testLoadMalformed)
  ]
}

/// Stores a g2o dataset as an array of guesses and measurements.
///
/// Used to test G2OReader.
struct G2OArray: G2OReader, Equatable {
  struct InitialGuess: Equatable {
    let index: Int
    let pose: Pose2
  }
  var initialGuesses: [InitialGuess] = []

  struct Measurement: Equatable {
    let frameIndex, measuredIndex: Int
    let pose: Pose2
  }
  var measurements: [Measurement] = []

  mutating func addInitialGuess(index: Int, pose: Pose2) {
    initialGuesses.append(InitialGuess(index: index, pose: pose))
  }
  mutating func addMeasurement(frameIndex: Int, measuredIndex: Int, pose: Pose2) {
    measurements.append(
      Measurement(frameIndex: frameIndex, measuredIndex: measuredIndex, pose: pose))
  }
}
