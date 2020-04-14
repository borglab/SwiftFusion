import SwiftFusion
import TensorFlow
import XCTest

final class GaussianFactorGraphTests: XCTestCase {
  
  func Vector2_t(_ x: Double, _ y: Double) -> Tensor<Double> {
    let t = Tensor<Double>(shape: [2, 1], scalars: [x, y])
    return t
  }
  
  /// Factor graph with 2 2D factors on 3 2D variables
  func createSimpleGaussianFactorGraph() -> GaussianFactorGraph {
    var fg = GaussianFactorGraph()
    
    let I_2x2: Tensor<Double> = eye(rowCount: 2)
    let x1 = 2, x2 = 0, l1 = 1;
    
    // linearized prior on x1: c[_x1_]+x1=0 i.e. x1=-c[_x1_]
    fg += JacobianFactor([x1], [10 * I_2x2], -1.0 * Vector2_t(1.0, 1.0))
    // odometry between x1 and x2: x2-x1=[0.2;-0.1]
    fg += JacobianFactor([x2, x1], [10 * I_2x2, -10 * I_2x2], Vector2_t(2.0, -1.0))
    // measurement between x1 and l1: l1-x1=[0.0;0.2]
    fg += JacobianFactor([l1, x1], [5 * I_2x2, -5 * I_2x2], Vector2_t(0.0, 1.0))
    // measurement between x2 and l1: l1-x2=[-0.2;0.3]
    fg += JacobianFactor([x2, l1], [-5 * I_2x2, 5 * I_2x2], Vector2_t(-1.0, 1.5))
    return fg;
  }

  /// test convergence for a simple Pose2SLAM
  func testTransposeMultiplication() {
    let A = createSimpleGaussianFactorGraph();

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
}
