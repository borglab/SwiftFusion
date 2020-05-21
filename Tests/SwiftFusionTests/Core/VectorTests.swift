import Foundation
import TensorFlow
import XCTest

import SwiftFusion

class VectorTests: XCTestCase {
  /// Tests that the initializers work.
  func testInitializers() {
    XCTAssertEqual(
      Vector([1, 2, 3]).scalars,
      [1, 2, 3]
    )
    XCTAssertEqual(
      Vector(zeros: 4).scalars,
      [0, 0, 0, 0]
    )
  }

  /// Tests that the miscellaneous computed properties work.
  func testComputedProperties() {
    XCTAssertEqual(
      Vector(zeros: 5).dimension,
      5
    )
  }

  /// Tests that the element arithmetic methods work.
  func testElementArithmetic() {
    let vector = Vector([1, 2, 3])
    XCTAssertEqual(vector.sum(), 6)
  }

  /// Test conversion to tensor.
  func testTensorConversion() {
    let vector = Vector([1, 2, 3])
    XCTAssertEqual(vector.tensor, Tensor<Double>([1, 2, 3]))
  }
}

/// Tests the `EuclideanVector` requirements.
class VectorEuclideanVectorTests: XCTestCase, EuclideanVectorTests {
  var dimension: Int { return 4 }

  var basisVectors: [Vector] {
    return (0..<dimension).map { index in
      var v = Array(repeating: Double(0), count: dimension)
      v[index] = 1
      return Vector(v)
    }
  }

  func makeVector(from start: Double, stride: Double) -> Vector {
    return Vector(Array((0..<dimension).map { start + Double($0) * stride }))
  }

  func testAll() {
    runAllTests()
  }
}
