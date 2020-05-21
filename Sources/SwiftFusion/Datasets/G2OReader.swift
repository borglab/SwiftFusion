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

import Foundation

/// Reads entries from a g2o dataset.
///
/// See https://lucacarlone.mit.edu/datasets/ for g2o specification and example datasets.
public protocol G2OReader {
  /// Creates an empty reader.
  init()

  /// Adds an initial guess that vertex `index` has pose `pose`.
  mutating func addInitialGuess(index: Int, pose: Pose2)

  /// Adds a measurement that vertex `measuredIndex` has relative pose `pose` in the frame of
  /// vertex `frameIndex`.
  ///
  /// TODO: Also read the information matrix for the measurement.
  mutating func addMeasurement(frameIndex: Int, measuredIndex: Int, pose: Pose2)
}

extension G2OReader {
  /// Creates a g2o dataset from `url`.
  public init(fromG2O url: URL) throws {
    self.init()
    try self.read(fromG2O: url)
  }

  /// Adds the g2o dataset at `url` into `self`.
  public mutating func read(fromG2O url: URL) throws {
    let lines = try String(contentsOf: url).split(separator: "\n")
    for (lineIndex, line) in lines.enumerated() {
      let columns = line.split(separator: " ")
      var columnIndex = 0

      /// Makes a `G2OParseError` referring to the current line.
      func makeG2OParseError(_ message: String) -> G2OParseError {
        return G2OParseError(lineIndex: lineIndex, line: String(line), message: message)
      }

      /// Returns the current `columnIndex` as a `String`, and advances the `columnIndex`.
      func parseString() -> Substring {
        let column = columns[columnIndex]
        columnIndex += 1
        return column
      }

      /// Returns the current `columnIndex` as an `Int`, and advances the `columnIndex`.
      func parseInt() throws -> Int {
        let column = parseString()
        guard let result = Int(column) else {
          throw makeG2OParseError("Cannot convert \(column) to Int")
        }
        return result
      }

      /// Returns the current `columnIndex` as an `Double`, and advances the `columnIndex`.
      func parseDouble() throws -> Double {
        let column = parseString()
        guard let result = Double(column) else {
          throw makeG2OParseError("Cannot convert \(column) to Double")
        }
        return result
      }

      /// Returns a `Vector2`, parsed starting at the current index, and advances the
      /// `columnIndex`.
      func parseVector2() throws -> Vector2 {
        return Vector2(try parseDouble(), try parseDouble())
      }

      /// Returns a `Pose2`, parsed starting at the current index, and advances the
      /// `columnIndex`.
      func parsePose2() throws -> Pose2 {
        let vec = try parseVector2()
        let rot = Rot2(try parseDouble())
        return Pose2(rot, vec)
      }

      let firstColumn = parseString()
      if firstColumn == "VERTEX_SE2" {
        // 1 "VERTEX_SE2" column + 1 index column + 3 Pose2 columns = 5
        guard columns.count == 5 else {
          throw makeG2OParseError(
            "VERTEX_SE2 row should have 5 columns, but it has \(columns.count)")
        }
        addInitialGuess(index: try parseInt(), pose: try parsePose2())
      } else if firstColumn == "EDGE_SE2" {
        // 1 "EDGE_SE2" column + 2 index columns + 3 Pose2 columns + 6 information matrix columns
        //   = 12
        guard columns.count == 12 else {
          throw makeG2OParseError(
            "EDGE_SE2 row should have 12 columns, but it has \(columns.count)")
        }
        addMeasurement(
          frameIndex: try parseInt(),
          measuredIndex: try parseInt(),
          pose: try parsePose2()
        )
        // Consume the information matrix columns.
        // TODO: Pass these to `addMeasurement`.
        for _ in 0..<6 {
          _ = try parseDouble()
        }
      } else {
        throw makeG2OParseError(
          "First column should be VERTEX_SE2 or EDGE_SE2, but it is \(firstColumn)")
      }

      assert(columnIndex == columns.count)
    }
  }
}

public struct G2OParseError: Swift.Error {
  public let lineIndex: Int
  public let line: String
  public let message: String
}
