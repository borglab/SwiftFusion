import Foundation
import TensorFlow
import XCTest

import SwiftFusion

class VectorTests: XCTestCase {
  static var allTests = [
    ("testInitializers", testInitializers),
    ("testComputedProperties", testComputedProperties),
    ("testElementArithmetic", testElementArithmetic),
    ("testEuclideanNorm", testEuclideanNorm),
    ("testAdditiveArithmetic", testAdditiveArithmetic),
    ("testVectorProtocol", testVectorProtocol),
    ("testTensorConversion", testTensorConversion),
  ]

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
    XCTAssertEqual(vector.squared(), Vector([1, 4, 9]))
  }

  /// Tests that the euclidean norm methods work.
  func testEuclideanNorm() {
    let vector = Vector([3, 4])
    XCTAssertEqual(vector.squaredNorm, 25)
    XCTAssertEqual(vector.norm, 5)
  }

  /// Test the AdditiveArithmetic conformance.
  func testAdditiveArithmetic() {
    let vector1 = Vector([1, 2, 3])
    let vector2 = Vector([4, 5, 6])
    XCTAssertEqual(Vector.zero, Vector([]))
    XCTAssertEqual(vector1 + vector2, Vector([5, 7, 9]))
    XCTAssertEqual(vector1 - vector2, Vector([-3, -3, -3]))
  }

  /// Test the VectorProtocol conformance.
  func testVectorProtocol() {
    let vector1 = Vector([1, 2, 3])
    XCTAssertEqual(vector1.adding(1), Vector([2, 3, 4]))
    XCTAssertEqual(vector1.subtracting(1), Vector([0, 1, 2]))
    XCTAssertEqual(vector1.scaled(by: 2), Vector([2, 4, 6]))
  }

  /// Test conversion to tensor.
  func testTensorConversion() {
    let vector = Vector([1, 2, 3])
    XCTAssertEqual(vector.tensor, Tensor<Double>([1, 2, 3]))
  }
}
