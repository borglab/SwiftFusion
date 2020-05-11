import Foundation
import TensorFlow
import XCTest

import SwiftFusion

class MatrixTests: XCTestCase {
  static var allTests = [
    ("testInitializers", testInitializers),
    ("testSubscripts", testSubscripts),
    ("testElementArithmetic", testElementArithmetic),
    ("testMutations", testMutations),
    ("testMatVec", testMatVec),
  ]

  /// Tests that the initializers work.
  func testInitializers() {
    let matrix1 = Matrix([1, 2, 3, 4, 5, 6], rowCount: 2, columnCount: 3)
    XCTAssertEqual(matrix1.scalars, [1, 2, 3, 4, 5, 6])
    XCTAssertEqual(matrix1.rowCount, 2)
    XCTAssertEqual(matrix1.columnCount, 3)

    let matrix2 = Matrix(eye: 3)
    XCTAssertEqual(matrix2.scalars, [1, 0, 0, 0, 1, 0, 0, 0, 1])
    XCTAssertEqual(matrix2.rowCount, 3)
    XCTAssertEqual(matrix2.columnCount, 3)

    let matrix3 = Matrix(stacking: [Vector([1, 2, 3]), Vector([4, 5, 6])])
    XCTAssertEqual(matrix3.scalars, [1, 2, 3, 4, 5, 6])
    XCTAssertEqual(matrix3.rowCount, 2)
    XCTAssertEqual(matrix3.columnCount, 3)
  }

  /// Tests that the subscripts work.
  func testSubscripts() {
    let matrix1 = Matrix([1, 2, 3, 4, 5, 6], rowCount: 2, columnCount: 3)
    XCTAssertEqual(matrix1[0, 0], 1)
    XCTAssertEqual(matrix1[0, 1], 2)
    XCTAssertEqual(matrix1[0, 2], 3)
    XCTAssertEqual(matrix1[1, 0], 4)
    XCTAssertEqual(matrix1[1, 1], 5)
    XCTAssertEqual(matrix1[1, 2], 6)
  }

  /// Tests that the element arithmetic methods work.
  func testElementArithmetic() {
    let matrix1 = Matrix([1, 2, 3, 4, 5, 6], rowCount: 2, columnCount: 3)
    XCTAssertEqual(2 * matrix1, Matrix([2, 4, 6, 8, 10, 12], rowCount: 2, columnCount: 3))
    XCTAssertEqual(matrix1 * 2, Matrix([2, 4, 6, 8, 10, 12], rowCount: 2, columnCount: 3))
  }

  /// Test that the mutation methods work.
  func testMutations() {
    var matrix1 = Matrix([], rowCount: 0, columnCount: 2)
    matrix1.append(row: Vector([1, 2]))
    XCTAssertEqual(matrix1, Matrix([1, 2], rowCount: 1, columnCount: 2))
    matrix1.append(row: Vector([3, 4]))
    XCTAssertEqual(matrix1, Matrix([1, 2, 3, 4], rowCount: 2, columnCount: 2))
  }

  /// Test matrix-vector multiplication.
  func testMatVec() {
    let matrix1 = Matrix([1, 2, 3, 4, 5, 6], rowCount: 2, columnCount: 3)
    let vector1 = Vector([10, 20, 30])
    let vector2 = Vector([40, 50])
    XCTAssertEqual(
      matvec(matrix1, vector1),
      Vector([140, 320])
    )
    XCTAssertEqual(
      matvec(matrix1, transposed: true, vector2),
      Vector([240, 330, 420])
    )
  }
}
