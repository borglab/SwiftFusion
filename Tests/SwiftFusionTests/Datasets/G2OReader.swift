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
  /// Tests loading a simple 2d g2o file using G2OReader.
  func testLoadSimple2D() {
    var dataset = G2OArray<Pose2>()
    try! G2OReader.read2D(file: dataDirectory.appendingPathComponent("simple2d.g2o")) {
      dataset.handleEntry($0)
    }
    let expectedDataset = G2OArray<Pose2>(
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

  /// Tests loading a simple 3d g2o file using G2OReader.
  func testLoadSimple3D() {
    /// Returns the expected `Pose3` that should be read from `g2oEntries`.
    func pose3(_ g2oEntries: [Double]) -> Pose3 {
      let s = g2oEntries
      let t = Vector3(s[0], s[1], s[2])
      let r = Rot3.fromQuaternion(s[6], s[3], s[4], s[5])
      return Pose3(r, t)
    }

    var dataset = G2OArray<Pose3>()
    try! G2OReader.read3D(file: dataDirectory.appendingPathComponent("simple3d.g2o")) {
      dataset.handleEntry($0)
    }
    let expectedDataset = G2OArray<Pose3>(
      initialGuesses: [
        G2OArray.InitialGuess(
          index: 0,
          pose: pose3([18.7381, 2.74428e-07, 98.2287, 0, 0, 0, 1])),
        G2OArray.InitialGuess(
          index: 1,
          pose: pose3([19.0477, 2.34636, 98.2319, -0.139007, 0.0806488, 0.14657, 0.976059])),
      ],
      measurements: [
        G2OArray.Measurement(
          frameIndex: 0,
          measuredIndex: 1,
          pose: pose3([0.309576, 2.34636, 0.00315914, -0.139007, 0.0806488, 0.14657, 0.976059]))
      ]
    )
    XCTAssertEqual(dataset, expectedDataset)
  }

  /// Tests that an error is thrown when reading a file that does not exist.
  func testLoadNotExist() {
    XCTAssertThrowsError(
      try G2OReader.read2D(file: dataDirectory.appendingPathComponent("notexist.g2o"), { _ in })
    )
  }

  /// Tests that an error is thrown when reading a malformed g2o file.
  func testLoadMalformed() {
    XCTAssertThrowsError(
      try G2OReader.read2D(file: dataDirectory.appendingPathComponent("notexist.g2o"), { _ in })
    )
  }

  /// Data directory for these tests.
  let dataDirectory = URL.sourceFileDirectory().appendingPathComponent("Data")
}

/// Stores a g2o dataset as an array of guesses and measurements.
///
/// Used to test G2OReader.
struct G2OArray<Pose: Equatable>: Equatable {
  struct InitialGuess: Equatable {
    let index: Int
    let pose: Pose
  }
  var initialGuesses: [InitialGuess] = []

  struct Measurement: Equatable {
    let frameIndex, measuredIndex: Int
    let pose: Pose
  }
  var measurements: [Measurement] = []

  mutating func handleEntry(_ entry: G2OReader.Entry<Pose>) {
    switch entry {
    case .initialGuess(index: let index, pose: let pose):
      initialGuesses.append(InitialGuess(index: index, pose: pose))
    case .measurement(frameIndex: let frameIndex, measuredIndex: let measuredIndex, pose: let pose):
      measurements.append(
        Measurement(frameIndex: frameIndex, measuredIndex: measuredIndex, pose: pose))
    }
  }
}
