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
  
  /// Tests vector bracket.
  func testVectorBracket() {
    let v1 = Vector2d(1, 2)
    let v2 = Vector2d(3, 4)
    XCTAssertEqual(v1.bracket(v2), 11)
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
  
  /// Tests the derivative of adding two vectors.
  func testVectorAddDerivative() {
    let v1 = Vector2d(1, 2)
    let v2 = Vector2d(3, 4)
    let (value, pb) = valueWithPullback(at: v1, v2) {
      $0 + $1
    }
    
    XCTAssertEqual(value, Vector2d(4, 6))
    XCTAssertEqual(pb(Vector2d(1, 0)).0, Vector2d(1, 0))
    XCTAssertEqual(pb(Vector2d(1, 0)).1, Vector2d(1, 0))
    
    XCTAssertEqual(pb(Vector2d(0, 1)).0, Vector2d(0, 1))
    XCTAssertEqual(pb(Vector2d(0, 1)).1, Vector2d(0, 1))
  }
  
  /// Tests the derivative of scaling a vector.
  func testVectorScaleDerivative() {
    let v1 = Vector2d(3, 4)
    let (value, pb) = valueWithPullback(at: 10, v1) {
      $0 * $1
    }
    
    XCTAssertEqual(value, Vector2d(30, 40))
    XCTAssertEqual(pb(Vector2d(1, 0)).0, 3)
    XCTAssertEqual(pb(Vector2d(1, 0)).1, Vector2d(10, 0))
    
    XCTAssertEqual(pb(Vector2d(0, 1)).0, 4)
    XCTAssertEqual(pb(Vector2d(0, 1)).1, Vector2d(0, 10))
  }
  
  /// Tests the derivative of vector bracket.
  func testVectorBracketDerivative() {
    let v1 = Vector2d(1, 2)
    let v2 = Vector2d(3, 4)
    let (value, grad) = valueWithGradient(at: v1, v2) {
      $0.differentiableBracket($1)
    }
    
    XCTAssertEqual(value, 11)
    XCTAssertEqual(grad.0, v2)
    XCTAssertEqual(grad.1, v1)
  }
}
