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

/// A 3x5 matrix, where each row is a `Tuple2<Vector2, Vector3>`
fileprivate typealias Matrix3x2_3x3 = FixedSizeMatrix<Array3<Tuple2<Vector2, Vector3>>>

class FixedSizeMatrixTests: XCTestCase {
  func testShape() {
    XCTAssertEqual(Matrix3.shape, Array2(3, 3))
    XCTAssertEqual(Matrix3x2.shape, Array2(3, 2))
    XCTAssertEqual(Matrix3x2_3x3.shape, Array2(3, 5))
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

  func testInitRows() {
    let m1 = Matrix3x2(rows: Vector2(1, 2), Vector2(10, 20), Vector2(100, 200))
    XCTAssertEqual(m1[0, 0], 1)
    XCTAssertEqual(m1[0, 1], 2)
    XCTAssertEqual(m1[1, 0], 10)
    XCTAssertEqual(m1[1, 1], 20)
    XCTAssertEqual(m1[2, 0], 100)
    XCTAssertEqual(m1[2, 1], 200)

    let (m2, pb) = valueWithPullback(at: 1) { x in
      Matrix3x2(rows: Vector2(1 * x, 2 * x), Vector2(10 * x, 20 * x), Vector2(100 * x, 200 * x))
    }
    XCTAssertEqual(m2[0, 0], 1)
    XCTAssertEqual(m2[0, 1], 2)
    XCTAssertEqual(m2[1, 0], 10)
    XCTAssertEqual(m2[1, 1], 20)
    XCTAssertEqual(m2[2, 0], 100)
    XCTAssertEqual(m2[2, 1], 200)
    XCTAssertEqual(pb(Matrix3x2([1, 0, 0, 0, 0, 0])), 1)
    XCTAssertEqual(pb(Matrix3x2([0, 1, 0, 0, 0, 0])), 2)
    XCTAssertEqual(pb(Matrix3x2([0, 0, 1, 0, 0, 0])), 10)
    XCTAssertEqual(pb(Matrix3x2([0, 0, 0, 1, 0, 0])), 20)
    XCTAssertEqual(pb(Matrix3x2([0, 0, 0, 0, 1, 0])), 100)
    XCTAssertEqual(pb(Matrix3x2([0, 0, 0, 0, 0, 1])), 200)
  }

  func testIdentity() {
    let m1 = Matrix3.identity
    for i in 0..<3 {
      for j in 0..<3 {
        XCTAssertEqual(m1[i, j], i == j ? 1 : 0)
      }
    }
  }

  func testInitMatrix3Helper() {
    let m1 = Matrix3(1, 2, 3, 4, 5, 6, 7, 8, 9)
    XCTAssertEqual(m1[0, 0], 1)
    XCTAssertEqual(m1[0, 1], 2)
    XCTAssertEqual(m1[0, 2], 3)
    XCTAssertEqual(m1[1, 0], 4)
    XCTAssertEqual(m1[1, 1], 5)
    XCTAssertEqual(m1[1, 2], 6)
    XCTAssertEqual(m1[2, 0], 7)
    XCTAssertEqual(m1[2, 1], 8)
    XCTAssertEqual(m1[2, 2], 9)
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

  func testAdd() {
    let m1 = Matrix3x2([0, 1, 2, 3, 4, 5])
    let m2 = Matrix3x2([0, 10, 20, 30, 40, 50])
    let expected = Matrix3x2([0, 11, 22, 33, 44, 55])
    XCTAssertEqual(m1 + m2, expected)

    let (value, pb) = valueWithPullback(at: m1, m2) { $0 + $1 }
    XCTAssertEqual(value, expected)
    for b in Matrix3x2.standardBasis {
      XCTAssertEqual(pb(b).0, b)
      XCTAssertEqual(pb(b).1, b)
    }
  }

  func testSubtract() {
    let m1 = Matrix3x2([0, 11, 22, 33, 44, 55])
    let m2 = Matrix3x2([0, 1, 2, 3, 4, 5])
    let expected = Matrix3x2([0, 10, 20, 30, 40, 50])
    XCTAssertEqual(m1 - m2, expected)

    let (value, pb) = valueWithPullback(at: m1, m2) { $0 - $1 }
    XCTAssertEqual(value, expected)
    for b in Matrix3x2.standardBasis {
      XCTAssertEqual(pb(b).0, b)
      XCTAssertEqual(pb(b).1, -b)
    }
  }

  func testMultiply() {
    let m = Matrix3x2([0, 1, 2, 3, 4, 5])
    let s: Double = 10
    let expected = Matrix3x2([0, 10, 20, 30, 40, 50])
    XCTAssertEqual(s * m, expected)

    let (value, pb) = valueWithPullback(at: s, m) { $0 * $1 }
    XCTAssertEqual(value, expected)
    for (i, b) in Matrix3x2.standardBasis.enumerated() {
      XCTAssertEqual(pb(b).0, Double(i))
      XCTAssertEqual(pb(b).1, s * b)
    }
  }

  func testDivide() {
    let m = Matrix3x2([0, 10, 20, 30, 40, 50])
    let s: Double = 10
    let expected = Matrix3x2([0, 1, 2, 3, 4, 5])
    XCTAssertEqual(m / s, expected)

    let (value, pb) = valueWithPullback(at: m, s) { $0 / $1 }
    XCTAssertEqual(value, expected)
    for (i, b) in Matrix3x2.standardBasis.enumerated() {
      XCTAssertEqual(pb(b).0, b / s)
      XCTAssertEqual(pb(b).1, -Double(i) / 10, accuracy: 1e-10)
    }
  }

  func testDot() {
    let m1 = Matrix2([1, 2, 3, 4])
    let m2 = Matrix2([1, 10, 100, 1000])
    XCTAssertEqual(m1.dot(m2), 4321)

    let (value, grad) = valueWithGradient(at: m1, m2) { $0.dot($1) }
    XCTAssertEqual(value, 4321)
    XCTAssertEqual(grad.0, m2)
    XCTAssertEqual(grad.1, m1)
  }

  func testDimension() {
    XCTAssertEqual(Matrix3.dimension, 9)
    XCTAssertEqual(Matrix3x2.dimension, 6)
    XCTAssertEqual(Matrix3x2_3x3.dimension, 15)
  }

  func testIsSquare() {
    XCTAssertEqual(Matrix3.isSquare, true)
    XCTAssertEqual(Matrix3x2.isSquare, false)
    XCTAssertEqual(Matrix3x2_3x3.isSquare, false)
  }

  func testOuterProduct() {
    let v1 = Vector2(1, 2)
    let v2 = Vector2(10, 100)
    XCTAssertEqual(
      Matrix2(outerProduct: v1, v2),
      Matrix2([
        10, 100,
        20, 200
      ])
    )
  }

  func testTransposed() {
    let m = Matrix3(
      0, 1, 2,
      3, 4, 5,
      6, 7, 8
    )
    let expected = Matrix3(
      0, 3, 6,
      1, 4, 7,
      2, 5, 8
    )
    XCTAssertEqual(m.transposed(), expected)

    let (value, pb) = valueWithPullback(at: m) { $0.transposed() }
    XCTAssertEqual(value, expected)
    for b in Matrix3.standardBasis {
      XCTAssertEqual(pb(b), b.transposed())
    }
  }

  func testMatVec() {
    let m = Matrix3(
      0, 1, 2,
      3, 4, 5,
      6, 7, 8
    )
    let v = Vector3(1, 10, 100)
    let expected = Vector3(210, 543, 876)
    XCTAssertEqual(matvec(m, v), expected)

    let (value, pb) = valueWithPullback(at: m, v) { matvec($0, $1) }
    XCTAssertEqual(value, expected)

    XCTAssertEqual(
      pb(Vector3(1, 0, 0)).0,
      Matrix3(
        1, 10, 100,
        0, 0, 0,
        0, 0, 0
      )
    )
    XCTAssertEqual(
      pb(Vector3(0, 1, 0)).0,
      Matrix3(
        0, 0, 0,
        1, 10, 100,
        0, 0, 0
      )
    )
    XCTAssertEqual(
      pb(Vector3(0, 0, 1)).0,
      Matrix3(
        0, 0, 0,
        0, 0, 0,
        1, 10, 100
      )
    )

    XCTAssertEqual(pb(Vector3(1, 0, 0)).1, Vector3(0, 1, 2))
    XCTAssertEqual(pb(Vector3(0, 1, 0)).1, Vector3(3, 4, 5))
    XCTAssertEqual(pb(Vector3(0, 0, 1)).1, Vector3(6, 7, 8))
  }

  func testMatMul() {
    let m1 = Matrix3(
      0, 1, 2,
      3, 4, 5,
      6, 7, 8
    )
    let m2 = Matrix3(
      1,   10,  100,
      10,  100, 1,
      100, 1,   10
    )
    let expected = Matrix3(
      210, 102, 021,
      543, 435, 354,
      876, 768, 687
    )
    XCTAssertEqual(matmul(m1, m2), expected)

    let (value, pb) = valueWithPullback(at: m1, m2) { matmul($0, $1) }
    XCTAssertEqual(value, expected)

    XCTAssertEqual(
      pb(Matrix3(
        1, 0, 0,
        0, 0, 0,
        0, 0, 0
      )).0,
      Matrix3(
        1, 10, 100,
        0, 0, 0,
        0, 0, 0
      )
    )

    XCTAssertEqual(
      pb(Matrix3(
        1, 0, 0,
        0, 0, 0,
        0, 0, 0
      )).1,
      Matrix3(
        0, 0, 0,
        1, 0, 0,
        2, 0, 0
      )
    )
  }

  func testKeyPathIterable() {
    let m1 = Matrix3x2([0, 1, 2, 3, 4, 5])
    XCTAssertEqual(
      m1.allKeyPaths(to: Double.self).map { m1[keyPath: $0] },
      [0, 1, 2, 3, 4, 5]
    )
  }

  func testCustomStringConvertible() {
    XCTAssertEqual(
      Matrix2([1, 2, 3, 4]).description,
      "Matrix(Vector2(x: 1.0, y: 2.0), Vector2(x: 3.0, y: 4.0))"
    )
  }
}
