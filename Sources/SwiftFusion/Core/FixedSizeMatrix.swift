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

import PenguinStructures

/// A matrix whose dimensions are known at compile time.
///
/// Stored as a fixed-size array of rows, where the rows are `EuclideanVectorN`. For example,
/// - `FixedSizeMatrix<Array3<Vector2>>`: A 3x2 matrix, where each row is a `Vector2`.
/// - `FixedSizeMatrix<Array3<Tuple2<Vector2, Vector3>>`: A 3x5 matrix, where each row is a
///   `Tuple2<Vector2, Vector3>`.
public struct FixedSizeMatrix<Rows: Equatable & FixedSizeArray> where Rows.Element: EuclideanVectorN {
  public var rows: Rows

  /// The count of rows.
  public static var rowCount: Int { return Rows.count }

  /// The count of columns.
  public static var columnCount: Int { return Rows.Element.dimension }

  /// Accesses the element at `row`, `column`.
  public subscript(row: Int, column: Int) -> Double {
    _read {
      yield rows[row][column]
    }
    _modify {
      yield &rows[row][column]
    }
  }
}

extension FixedSizeMatrix: Equatable {}

extension FixedSizeMatrix {
  /// Creates a matrix with `elements`.
  ///
  /// Requires: `elements.count == Self.rowCount * Self.columnCount`.
  public init<C: Collection>(_ elements: C) where C.Element == Double {
    self.rows = Rows((0..<Rows.count).lazy.map { i in
      Rows.Element(elements.dropFirst(i * Self.columnCount))
    })
  }

  /// The identity matrix.
  ///
  /// - Requires: `Self` is a square matrix. e.g. `Self.rowCount == Self.columnCount`.
  public static var identity: Self {
    precondition(Self.rowCount == Self.columnCount)
    var r = Self.zero
    for i in 0..<Self.rowCount {
      r[i, i] = 1
    }
    return r
  }
}

extension FixedSizeMatrix: AdditiveArithmetic {
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    var result = lhs
    for i in 0..<Self.rowCount {
      for j in 0..<Self.columnCount {
        result[i, j] += rhs[i, j]
      }
    }
    return result
  }

  public static func - (_ lhs: Self, _ rhs: Self) -> Self {
    var result = lhs
    for i in 0..<Self.rowCount {
      for j in 0..<Self.columnCount {
        result[i, j] -= rhs[i, j]
      }
    }
    return result
  }

  public static var zero: Self {
    return Self(rows: Rows((0..<Rows.count).lazy.map { _ in Rows.Element.zero }))
  }
}
