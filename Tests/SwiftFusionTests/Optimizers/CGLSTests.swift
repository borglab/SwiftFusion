// This file tests for the CGLS optimizer

import SwiftFusion
import PenguinStructures
import TensorFlow
import XCTest

final class CGLSTests: XCTestCase {  
  /// test convergence for a simple gaussian factor graph
  func testCGLSSolver() {
    let gfg = SimpleGaussianFactorGraph.create()
    
    let optimizer = CGLS(precision: 1e-7, max_iteration: 10)
    var x: VectorValues = SimpleGaussianFactorGraph.zeroDelta()
    optimizer.optimize(gfg: gfg, initial: &x)
    
    let expected = SimpleGaussianFactorGraph.correctDelta()
    
    for k in x.keys {
      assertEqual(x[k].tensor, expected[k].tensor, accuracy: 1e-6)
    }
  }
  
  /// test to ensure we are matching the GTSAM behavior
  func testSanity() {
    var v_linear = VariableAssignments()
    let v0 = v_linear.store(Vector2(0, 0))
    let v1 = v_linear.store(Vector2(0, 0))
    let v2 = v_linear.store(Vector2(0, 0))
    
    var linearGraph = GaussianFactorGraph(zeroValues: v_linear.tangentVectorZeros)
    
    linearGraph.store(JacobianFactor(jacobian: Array2([
      Tuple1(Vector2(10, 0)),
      Tuple1(Vector2(0, 10))
    ]), error: Vector2(-1, -1), edges: Tuple1(v2)))
//    linearGraph.store(JacobianFactor(2, (Matrix(2,2)<< -10, 0, 0, -10).finished(), 0, (Matrix(2,2)<< 10, 0, 0, 10).finished(), (Vector(2) << 2, -1).finished(), unit2);
    linearGraph.store(JacobianFactor(jacobian: Array2([
      Tuple2(Vector2(-10, 0), Vector2(10, 0)),
      Tuple2(Vector2(0, -10), Vector2(0, 10))
    ]), error: Vector2(2, -1), edges: Tuple2(v2, v0)))
//    linearGraph.store(JacobianFactor(2, (Matrix(2,2)<< -5, 0, 0, -5).finished(), 1, (Matrix(2,2)<< 5, 0, 0, 5).finished(), (Vector(2) << 0, 1).finished(), unit2);
    linearGraph.store(JacobianFactor(jacobian: Array2([
      Tuple2(Vector2(-5, 0), Vector2(5, 0)),
      Tuple2(Vector2(0, -5), Vector2(0, 5))
    ]), error: Vector2(0, 1), edges: Tuple2(v2, v1)))
//    linearGraph.store(JacobianFactor(0, (Matrix(2,2)<< -5, 0, 0, -5).finished(), 1, (Matrix(2,2)<< 5, 0, 0, 5).finished(), (Vector(2) << -1, 1.5).finished(), unit2);
    linearGraph.store(JacobianFactor(jacobian: Array2([
      Tuple2(Vector2(-5, 0), Vector2(5, 0)),
      Tuple2(Vector2(0, -5), Vector2(0, 5))
    ]), error: Vector2(-1, 1.5), edges: Tuple2(v0, v1)))
//    linearGraph.store(JacobianFactor(0, (Matrix(2,2)<< 1, 0, 0, 1).finished(), (Vector(2) << 0, 0).finished(), unit2);
    linearGraph.store(JacobianFactor(jacobian: Array2([
      Tuple1(Vector2(1, 0)),
      Tuple1(Vector2(0, 1))
    ]), error: Vector2(0, 0), edges: Tuple1(v0)))
//    linearGraph.store(JacobianFactor(1, (Matrix(2,2)<< 1, 0, 0, 1).finished(), (Vector(2) << 0, 0).finished(), unit2);
    linearGraph.store(JacobianFactor(jacobian: Array2([
      Tuple1(Vector2(1, 0)),
      Tuple1(Vector2(0, 1))
    ]), error: Vector2(0, 0), edges: Tuple1(v1)))
//    linearGraph.store(JacobianFactor(2, (Matrix(2,2)<< 1, 0, 0, 1).finished(), (Vector(2) << 0, 0).finished(), unit2);
    linearGraph.store(JacobianFactor(jacobian: Array2([
      Tuple1(Vector2(1, 0)),
      Tuple1(Vector2(0, 1))
    ]), error: Vector2(0, 0), edges: Tuple1(v2)))
    
    var r = linearGraph.errorVectors(at: v_linear) // r(0) = b - A * x(0), the residual
    var p = linearGraph.errorVectors_linearComponent_adjoint(r) // p(0) = s(0) = A^T * r(0), residual in value space
    var s = p // residual of normal equations
    var gamma = s.squaredNorm // γ(0) = ||s(0)||^2

    let q = linearGraph.errorVectors_linearComponent(at: p) // q(k) = A * p(k)
    print("""
Error1 = \((0..<3).map { r[$0, factorType: JacobianFactor<Array2<Tuple2<Vector2, Vector2>>, Vector2>.self] })
""")
    print("""
    Error2 = \((0..<3).map { r[$0, factorType: JacobianFactor<Array2<Tuple1<Vector2>>, Vector2>.self] })
    """)
    // Error1 = [SwiftFusion.Vector2(x: 2.0, y: -1.0), SwiftFusion.Vector2(x: 0.0, y: 1.0), SwiftFusion.Vector2(x: -1.0, y: 1.5)]
//    Error2 = [SwiftFusion.Vector2(x: -1.0, y: -1.0), SwiftFusion.Vector2(x: 0.0, y: 0.0), SwiftFusion.Vector2(x: 0.0, y: 0.0)]
    // error = [SwiftFusion.Vector(scalarsStorage: [1.0, 1.0]), SwiftFusion.Vector(scalarsStorage: [-2.0, 1.0]), SwiftFusion.Vector(scalarsStorage: [0.0, -1.0]), SwiftFusion.Vector(scalarsStorage: [1.0, -1.5]), SwiftFusion.Vector(scalarsStorage: [0.0, 0.0]), SwiftFusion.Vector(scalarsStorage: [0.0, 0.0]), SwiftFusion.Vector(scalarsStorage: [0.0, 0.0])]
    // NOTE: Should be [25; -17.5; -5; 12.5; -30; -5];
  }
}
