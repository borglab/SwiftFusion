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
}

/// Tests the `EuclideanVector` requirements.
class VectorEuclideanVectorTests: XCTestCase, EuclideanVectorTests {
  typealias Testee = Vector
  static var dimension: Int { return 4 }
  func testAll() {
    runAllEuclideanVectorTests()
  }
}
