import SwiftFusion
import TensorFlow
import XCTest

final class OldGaussianFactorGraphTests: XCTestCase {  
  /// test ATr
  func testTransposeMultiplication() {
    let A = SimpleGaussianFactorGraph.create()

    var e = Errors()
    e += [Vector([0.0, 0.0])]
    e += [Vector([15.0, 0.0])]
    e += [Vector([0.0, -5.0])]
    e += [Vector([-7.5, -5.0])]

    var expected = VectorValues()
    expected.insert(1, Vector([-37.5, -50.0]))
    expected.insert(2, Vector([-150.0, 25.0]))
    expected.insert(0, Vector([187.5, 25.0]))

    let actual = A.atr(e)
    XCTAssertEqual(expected, actual)
  }
  
  /// test Ax
  func testMultiplication() {
    let A = SimpleGaussianFactorGraph.create()

    var expected = Errors()
    expected += [Vector([-1.0, -1.0])]
    expected += [Vector([2.0, -1.0])]
    expected += [Vector([0.0, 1.0])]
    expected += [Vector([-1.0, 1.5])]

    let x = SimpleGaussianFactorGraph.correctDelta()

    let actual = A * x
    XCTAssertEqual(expected, actual)
  }
}
