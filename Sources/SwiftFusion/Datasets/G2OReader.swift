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

/// Namespace for g2o reader types and functions.
///
/// See https://lucacarlone.mit.edu/datasets/ for g2o specification and example datasets.
public enum G2OReader {
  /// A G2O problem expressed as a `NonlinearFactorGraph`.
  public struct G2ONonlinearFactorGraph {
    /// The initial guess.
    public var initialGuess: Values = Values()

    /// The factor graph representing the measurements.
    public var graph: NonlinearFactorGraph = NonlinearFactorGraph()

    /// Creates a problem from the given 2D file.
    public init(g2oFile2D: URL) throws {
      try G2OReader.read2D(file: g2oFile2D) { self.handleEntry($0) }
    }

    /// Creates a problem from the given 3D file.
    public init(g2oFile3D: URL) throws {
      try G2OReader.read3D(file: g2oFile3D) { self.handleEntry($0) }
    }

    private mutating func handleEntry<Pose: LieGroup>(_ entry: Entry<Pose>) {
      switch entry {
      case .initialGuess(index: let index, pose: let pose):
        initialGuess.insert(index, pose)
      case .measurement(frameIndex: let frameIndex, measuredIndex: let measuredIndex, pose: let pose):
        graph += BetweenFactor(frameIndex, measuredIndex, pose)
      }
    }
  }

  /// An entry in a G2O file.
  public enum Entry<Pose> {
    /// An initial guess that vertex `index` has pose `pose`.
    case initialGuess(index: Int, pose: Pose)

    /// A measurement that vertex `measuredIndex` has relative pose `pose` in the frame of
    /// vertex `frameIndex`.
    ///
    /// TODO: Include the information matrix for the measurement.
    case measurement(frameIndex: Int, measuredIndex: Int, pose: Pose)
  }

  /// Calls `handleEntry` for each entry in `file`.
  public static func read2D(file: URL, _ handleEntry: (Entry<Pose2>) -> ()) throws {
    let lines = try String(contentsOf: file).split(separator: "\n")
    for (lineIndex, line) in lines.enumerated() {
      var parser = G2OLineParser(line, index: lineIndex)
      let firstColumn = try parser.parseString()
      if firstColumn == "VERTEX_SE2" {
        handleEntry(.initialGuess(index: try parser.parseInt(), pose: try parser.parsePose2()))
      } else if firstColumn == "EDGE_SE2" {
        handleEntry(.measurement(
          frameIndex: try parser.parseInt(),
          measuredIndex: try parser.parseInt(),
          pose: try parser.parsePose2()
        ))
        // Consume the information matrix columns.
        // TODO: Read these too.
        for _ in 0..<6 {
          _ = try parser.parseDouble()
        }
      } else {
        throw parser.makeParseError(
          "First column should be VERTEX_SE2 or EDGE_SE2, but it is \(firstColumn)")
      }
      try parser.checkConsumedAll()
    }
  }

  /// Calls `handleEntry` for each entry in `file`.
  public static func read3D(file: URL, _ handleEntry: (Entry<Pose3>) -> ()) throws {
    let lines = try String(contentsOf: file).split(separator: "\n")
    for (lineIndex, line) in lines.enumerated() {
      var parser = G2OLineParser(line, index: lineIndex)
      let firstColumn = try parser.parseString()
      if firstColumn == "VERTEX_SE3:QUAT" {
        handleEntry(.initialGuess(index: try parser.parseInt(), pose: try parser.parsePose3()))
      } else if firstColumn == "EDGE_SE3:QUAT" {
        handleEntry(.measurement(
          frameIndex: try parser.parseInt(),
          measuredIndex: try parser.parseInt(),
          pose: try parser.parsePose3()
        ))
        // Consume the information matrix columns.
        // TODO: Read these too.
        for _ in 0..<21 {
          _ = try parser.parseDouble()
        }
      } else {
        throw parser.makeParseError(
          "First column should be VERTEX_SE3:QUAT or EDGE_SE3:QUAT, but it is \(firstColumn)")
      }
      try parser.checkConsumedAll()
    }
  }

  public struct ParseError: Swift.Error {
    public let lineIndex: Int
    public let line: String
    public let message: String
  }
}

/// Stateful G2O line parser.
fileprivate struct G2OLineParser {
  private let line: Substring
  private let lineIndex: Int
  private var columns: [Substring]
  private var columnIndex: Int

  init(_ line: Substring, index: Int) {
    self.line = line
    self.lineIndex = index
    self.columns = line.split(separator: " ")
    self.columnIndex = 0
  }

  /// Makes a `G2OReader.ParseError` referring to the current line.
  func makeParseError(_ message: String) -> G2OReader.ParseError {
    return G2OReader.ParseError(lineIndex: lineIndex, line: String(line), message: message)
  }

  /// Returns the current `columnIndex` as a `String`, and advances the `columnIndex`.
  mutating func parseString() throws -> Substring {
    if columnIndex == columns.count {
      throw makeParseError("Fewer columns than expected")
    }
    let column = columns[columnIndex]
    columnIndex += 1
    return column
  }

  /// Returns the current `columnIndex` as an `Int`, and advances the `columnIndex`.
  mutating func parseInt() throws -> Int {
    let column = try parseString()
    guard let result = Int(column) else {
      throw makeParseError("Cannot convert \(column) to Int")
    }
    return result
  }

  /// Returns the current `columnIndex` as an `Double`, and advances the `columnIndex`.
  mutating func parseDouble() throws -> Double {
    let column = try parseString()
    guard let result = Double(column) else {
      throw makeParseError("Cannot convert \(column) to Double")
    }
    return result
  }

  /// Returns a `Vector2`, parsed starting at the current index, and advances the
  /// `columnIndex`.
  mutating func parseVector2() throws -> Vector2 {
    return Vector2(try parseDouble(), try parseDouble())
  }

  /// Returns a `Vector3`, parsed starting at the current index, and advances the
  /// `columnIndex`.
  mutating func parseVector3() throws -> Vector3 {
    return Vector3(try parseDouble(), try parseDouble(), try parseDouble())
  }

  /// Returns a `Rot3`, parsed as a quaternion starting at the current index, and advances the
  /// `columnIndex`.
  mutating func parseQuaternion() throws -> Rot3 {
    print("warning, totally incorrect quaternion parsing")
    let w = try parseDouble()
    let x = try parseDouble()
    let y = try parseDouble()
    let z = try parseDouble()
    return Rot3(w, x, y, z, 0, 0, 0, 0, 0)
  }

  /// Returns a `Pose2`, parsed starting at the current index, and advances the
  /// `columnIndex`.
  mutating func parsePose2() throws -> Pose2 {
    let vec = try parseVector2()
    let rot = Rot2(try parseDouble())
    return Pose2(rot, vec)
  }

  /// Returns a `Pose3`, parsed starting at the current index, and advances the
  /// `columnIndex`.
  mutating func parsePose3() throws -> Pose3 {
    let vec = try parseVector3()
    let rot = try parseQuaternion()
    return Pose3(rot, vec)
  }

  /// Throws a parse error if we have not consumed all the columns.
  func checkConsumedAll() throws {
    if columnIndex < columns.count {
      throw makeParseError("More columns than expected")
    }
  }
}
