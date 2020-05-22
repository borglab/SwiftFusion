import SwiftFusion
import XCTest

/// Tests that `BearingRangeError` satisfies the `EuclideanVectorN` requirements.
final class BearingRangeErrorEuclideanVectorNTests: XCTestCase, EuclideanVectorTests {
  typealias Testee = BearingRangeError<Vector3>
  static var dimension: Int { return 4 }
  func testAll() {
    runAllEuclideanVectorNTests()
  }
}
