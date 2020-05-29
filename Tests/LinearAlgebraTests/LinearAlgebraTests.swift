import LinearAlgebra
import XCTest

final class LinearAlgebraTests: XCTestCase {
  /// Tests adding two vectors.
  func testVectorAdd() {
    let v1 = Vector(1, 2)
    let v2 = Vector(3, 4)
    XCTAssertEqual(v1 + v2, Vector(4, 6))
  }

  /// Tests scaling a vector by a scalar.
  func testVectorScale() {
    let v1 = Vector(1, 2)
    XCTAssertEqual(3 * v1, Vector(3, 6))
  }

  /// Tests applying a matrix to a vector.
  func testMatrixVector() {
    let mat = Matrix(
      Vector(1, 0, 1),
      Vector(0, 1, 1)
    )
    let v = Vector(2, 3)
    XCTAssertEqual(mat * v, Vector(2, 3, 5))
  }

  /// Tests multiplying two matrices.
  func testMatmul() {
    let mat1 = Matrix(
      Vector(1, 0, 1),
      Vector(0, 1, 1)
    )
    let mat2 = Matrix(
      Vector(2, 3),
      Vector(10, 10)
    )

    let expectedProduct = Matrix(
      Vector(2, 3, 5),
      Vector(10, 10, 20)
    )

    XCTAssertEqual(mat1 * mat2, expectedProduct)
  }
}
