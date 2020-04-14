import SwiftFusion
import TensorFlow
import XCTest

final class GaussianFactorGraphTests: XCTestCase {  
  /// test ATr
  func testTransposeMultiplication() {
    let A = SimpleGaussianFactorGraph.create()

    var e = Errors()
    e += [Vector2_t(0.0, 0.0)]
    e += [Vector2_t(15.0, 0.0)]
    e += [Vector2_t(0.0, -5.0)]
    e += [Vector2_t(-7.5, -5.0)]

    var expected = VectorValues()
    expected.insert(1, Vector2_t(-37.5, -50.0))
    expected.insert(2, Vector2_t(-150.0, 25.0))
    expected.insert(0, Vector2_t(187.5, 25.0))

    let actual = A.atr(e)
    XCTAssertEqual(expected, actual)
  }
  
  /// test Ax
  func testMultiplication() {
    let A = SimpleGaussianFactorGraph.create()

    var expected = Errors()
    expected += [Vector2_t(-1.0, -1.0)]
    expected += [Vector2_t(2.0, -1.0)]
    expected += [Vector2_t(0.0, 1.0)]
    expected += [Vector2_t(-1.0, 1.5)]

    let x = SimpleGaussianFactorGraph.correctDelta()

    let actual = A * x
    XCTAssertEqual(expected, actual)
  }
}
