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
  
  /// test to ensure we are matching the GTSAM behavior
    func testSanity() {
      var v_linear = VectorValues()
      v_linear.insert(0, Vector([0, 0]))
      v_linear.insert(1, Vector([0, 0]))
      v_linear.insert(2, Vector([0, 0]))
      
      let I_2x2 = Matrix(eye: 2)
      
      var linearGraph = OldGaussianFactorGraph()
      // JacobianFactor(2, (Matrix(2,2)<< 10, 0, 0, 10).finished(), (Vector(2) << -1, -1).finished(), unit2);
      linearGraph += OldJacobianFactor([2], [10 * I_2x2], Vector([-1, -1]))
  //    linearGraph.store(JacobianFactor(2, (Matrix(2,2)<< -10, 0, 0, -10).finished(), 0, (Matrix(2,2)<< 10, 0, 0, 10).finished(), (Vector(2) << 2, -1).finished(), unit2);
      linearGraph += OldJacobianFactor([2, 0], [-10 * I_2x2, 10 * I_2x2], Vector([2, -1]))
  //    linearGraph.store(JacobianFactor(2, (Matrix(2,2)<< -5, 0, 0, -5).finished(), 1, (Matrix(2,2)<< 5, 0, 0, 5).finished(), (Vector(2) << 0, 1).finished(), unit2);
      linearGraph += OldJacobianFactor([2, 1], [-5 * I_2x2, 5 * I_2x2], Vector([0, 1]))
  //    linearGraph.store(JacobianFactor(0, (Matrix(2,2)<< -5, 0, 0, -5).finished(), 1, (Matrix(2,2)<< 5, 0, 0, 5).finished(), (Vector(2) << -1, 1.5).finished(), unit2);
      linearGraph += OldJacobianFactor([0, 1], [-5 * I_2x2, 5 * I_2x2], Vector([-1, 1.5]))
  //    linearGraph.store(JacobianFactor(0, (Matrix(2,2)<< 1, 0, 0, 1).finished(), (Vector(2) << 0, 0).finished(), unit2);
      linearGraph += OldJacobianFactor([0], [1 * I_2x2], Vector([0, 0]))
  //    linearGraph.store(JacobianFactor(1, (Matrix(2,2)<< 1, 0, 0, 1).finished(), (Vector(2) << 0, 0).finished(), unit2);
      linearGraph += OldJacobianFactor([1], [1 * I_2x2], Vector([0, 0]))
  //    linearGraph.store(JacobianFactor(2, (Matrix(2,2)<< 1, 0, 0, 1).finished(), (Vector(2) << 0, 0).finished(), unit2);
      linearGraph += OldJacobianFactor([2], [1 * I_2x2], Vector([0, 0]))
      
      print("error = \(linearGraph.residual(v_linear))")
      
      // NOTE: Should be [25; -17.5; -5; 12.5; -30; -5];
    }
}
