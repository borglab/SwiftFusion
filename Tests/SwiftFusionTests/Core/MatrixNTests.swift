import SwiftFusion
import XCTest

class MatrixNTests: XCTestCase {
  /// Test matrix-matrix multiplication.
  func testMatMul() {
    let matrix1 = Matrix3(1, 0, 0, 0, 1, 0, 0, 0, 1)
    
    XCTAssertEqual(
      matmul(matrix1, matrix1) as Matrix3,
      matrix1 as Matrix3
    )
    
    let matrix4 = Matrix3(5, 1 ,3,
     1, 1 , 1,
     1, 2 , 1)
    let matrix5 = Vector3(1, 2, 3)
    
    XCTAssertEqual(
      matvec(matrix4, matrix5),
      Vector3(16, 6, 8)
    )
  }
}
