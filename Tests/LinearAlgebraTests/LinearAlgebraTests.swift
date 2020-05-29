import LinearAlgebra
import XCTest

final class LinearAlgebraTests: XCTestCase {
  /// Tests adding two vectors.
  func testVectorAdd() {
    let v1 = Vector2d(1, 2)
    let v2 = Vector2d(3, 4)
    XCTAssertEqual(v1 + v2, Vector2d(4, 6))
  }
  
  /// Tests scaling a vector by a scalar.
  func testVectorScale() {
    let v1 = Vector2d(1, 2)
    XCTAssertEqual(3 * v1, Vector2d(3, 6))
  }
    
  /// Tests applying a matrix to a vector.
  func testMatrixVector() {
    let mat = Matrix3x2d(
      Vector3d(1, 0, 1),
      Vector3d(0, 1, 1)
    )
    let v = Vector2d(2, 3)
    XCTAssertEqual(mat(v), Vector3d(2, 3, 5))
  }
  
  /// Tests multiplying two matrices.
  func testMatmul() {
    let mat1 = Matrix3x2d(
      Vector3d(1, 0, 1),
      Vector3d(0, 1, 1)
    )
    let mat2 = Matrix2d(
      Vector2d(2, 3),
      Vector2d(10, 10)
    )

    let expectedProduct = Matrix3x2d(
      Vector3d(2, 3, 5),
      Vector3d(10, 10, 20)
    )
    
    XCTAssertEqual(mat1.matmul(mat2), expectedProduct)
  }
}
