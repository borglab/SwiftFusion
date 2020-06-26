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

import XCTest

import PenguinStructures
import SwiftFusion

/// A 3x2 matrix, where each row is a `Vector2`.
fileprivate typealias Matrix3x2 = FixedSizeMatrix<Array3<Vector2>>

/// A 3x3 matrix, where each row is a `Vector3`.
fileprivate typealias Matrix3x3 = FixedSizeMatrix<Array3<Vector3>>

/// A 3x5 matrix, where each row is a `Tuple2<Vector2, Vector3>`
fileprivate typealias Matrix3x2_3x3 = FixedSizeMatrix<Array3<Tuple2<Vector2, Vector3>>>

class FixedSizeMatrixTests: XCTestCase {
  func testRowColumnCount() {
    XCTAssertEqual(Matrix3x2.rowCount, 3)
    XCTAssertEqual(Matrix3x2.columnCount, 2)
    XCTAssertEqual(Matrix3x2_3x3.rowCount, 3)
    XCTAssertEqual(Matrix3x2_3x3.columnCount, 5)
  }

  func testZero() {
    let m1 = Matrix3x2.zero
    for i in 0..<3 {
      for j in 0..<2 {
        XCTAssertEqual(m1[i, j], 0)
      }
    }

    let m2 = Matrix3x2_3x3.zero
    for i in 0..<3 {
      for j in 0..<5 {
        XCTAssertEqual(m2[i, j], 0)
      }
    }
  }

  func testIdentity() {
    let m1 = Matrix3x3.identity
    for i in 0..<3 {
      for j in 0..<3 {
        XCTAssertEqual(m1[i, j], i == j ? 1 : 0)
      }
    }
  }

  func testInitElements() {
    let m1 = Matrix3x2([
      1, 2,
      10, 20,
      100, 200
    ])
    XCTAssertEqual(m1[0, 0], 1)
    XCTAssertEqual(m1[0, 1], 2)
    XCTAssertEqual(m1[1, 0], 10)
    XCTAssertEqual(m1[1, 1], 20)
    XCTAssertEqual(m1[2, 0], 100)
    XCTAssertEqual(m1[2, 1], 200)
  }

  func testAdd() {
    let m1 = Matrix3x2([0, 1, 2, 3, 4, 5])
    let m2 = Matrix3x2([0, 10, 20, 30, 40, 50])
    let expected = Matrix3x2([0, 11, 22, 33, 44, 55])
    XCTAssertEqual(m1 + m2, expected)
  }

  func testSubtract() {
    let m1 = Matrix3x2([0, 11, 22, 33, 44, 55])
    let m2 = Matrix3x2([0, 1, 2, 3, 4, 5])
    let expected = Matrix3x2([0, 10, 20, 30, 40, 50])
    XCTAssertEqual(m1 - m2, expected)
  }

}
