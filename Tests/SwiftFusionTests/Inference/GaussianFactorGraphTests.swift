// Copyright 2020 The SwiftFusion Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation
import TensorFlow
import XCTest

import PenguinStructures
import SwiftFusion

class GaussianFactorGraphTests: XCTestCase {
  /// Stores a factor in the graph and evaluates the graph error vector.
  func testStoreFactors() {
    var x = VariableAssignments()
    let id1 = x.store(Vector1(1))
    let id2 = x.store(Vector2(2, 3))

    let A = Array2(
      Tuple2(Vector1(10), Vector2(100, 1000)),
      Tuple2(Vector1(-10), Vector2(-100, -1000))
    )
    let b = Vector2(8, 9)

    var graph = GaussianFactorGraph(zeroValues: x.tangentVectorZeros)
    let f1 = JacobianFactor(jacobian: A, error: b, edges: Tuple2(id1, id2))
    graph.store(f1)

    // `b - A * x`
    let expected = Vector2(
      3210 - 8,
      -3210 - 9
    )
    XCTAssertEqual(graph.errorVectors(at: x)[0, factorType: type(of: f1)], expected)
  }

  /// Adds some scalar jacobians and tests that the resulting graph produces the correct error
  /// vectors.
  func testAddScalarJacobians() {
    var x = VariableAssignments()
    let id1 = x.store(Vector2(1, 2))
    let id2 = x.store(Vector2(3, 4))
    let id3 = x.store(Vector3(5, 6, 7))

    var jacobians = GaussianFactorGraph(zeroValues: x.tangentVectorZeros)
    jacobians.addScalarJacobians(10)

    let errorVectors = jacobians.errorVectors(at: x)
    XCTAssertEqual(
      errorVectors[0, factorType: ScalarJacobianFactor<Vector2>.self], Vector2(10, 20))
    XCTAssertEqual(
      errorVectors[1, factorType: ScalarJacobianFactor<Vector2>.self], Vector2(30, 40))
    XCTAssertEqual(
      errorVectors[0, factorType: ScalarJacobianFactor<Vector3>.self], Vector3(50, 60, 70))

    let linearComponent = jacobians.errorVectors_linearComponent(at: x)
    XCTAssertEqual(
      linearComponent[0, factorType: ScalarJacobianFactor<Vector2>.self], Vector2(10, 20))
    XCTAssertEqual(
      linearComponent[1, factorType: ScalarJacobianFactor<Vector2>.self], Vector2(30, 40))
    XCTAssertEqual(
      linearComponent[0, factorType: ScalarJacobianFactor<Vector3>.self], Vector3(50, 60, 70))

    let linearComponent_adjoint =
      jacobians.errorVectors_linearComponent_adjoint(linearComponent)
    XCTAssertEqual(linearComponent_adjoint[id1], Vector2(100, 200))
    XCTAssertEqual(linearComponent_adjoint[id2], Vector2(300, 400))
    XCTAssertEqual(linearComponent_adjoint[id3], Vector3(500, 600, 700))
  }
  
  func testJacobianFactorSanity() {
    var x = VariableAssignments()
    let id0 = x.store(Vector3(1.0, 1.0, 1.0))
    let id1 = x.store(Vector3(0.5, 0.5, 0.5))
    let id2 = x.store(Vector3(1.0/3, 1.0/3, 1.0/3))
    
    // let terms = [Matrix3.identity, 2 * Matrix3.identity, 3 * Matrix3.identity]
    let jf = JacobianFactor<Array3<Tuple3<Vector3, Vector3, Vector3>>, Vector3>(
      jacobians: FixedSizeMatrix3.identity, 2 * FixedSizeMatrix3.identity, 3 * FixedSizeMatrix3.identity,
      error: Vector3(1, 2, 3), edges: Tuple3(id0, id1, id2))
    
    var gfg = GaussianFactorGraph(zeroValues: x)
    gfg.store(jf)
    
    let r = gfg.errorVectors(at: x)
    
    assertAllKeyPathEqual(r[0, factorType: JacobianFactor<Array3<Tuple3<Vector3, Vector3, Vector3>>, Vector3>.self],
                          Vector3(2, 1, 0), accuracy: 1e-10)
  }

  /// Test Ax.
  func testMultiplication() {
    let A = SimpleGaussianFactorGraph.create()
    let Ax = A.errorVectors_linearComponent(at: SimpleGaussianFactorGraph.correctDelta())
    XCTAssertEqual(Ax[0, factorType: JacobianFactor2x2_1.self], Vector2(-1, -1))
    XCTAssertEqual(Ax[0, factorType: JacobianFactor2x2_2.self], Vector2(2, -1))
    XCTAssertEqual(Ax[1, factorType: JacobianFactor2x2_2.self], Vector2(0, 1))
    XCTAssertEqual(Ax[2, factorType: JacobianFactor2x2_2.self], Vector2(-1, 1.5))
  }

  /// Test A^Ty
  func testTransposeMultiplication() {
    var y = AllVectors()
    y.store(Vector2(0, 0), factorType: JacobianFactor2x2_1.self)
    y.store(Vector2(15, 0), factorType: JacobianFactor2x2_2.self)
    y.store(Vector2(0, -5), factorType: JacobianFactor2x2_2.self)
    y.store(Vector2(-7.5, -5), factorType: JacobianFactor2x2_2.self)
    let A = SimpleGaussianFactorGraph.create()
    let ATy = A.errorVectors_linearComponent_adjoint(y)
    XCTAssertEqual(ATy[SimpleGaussianFactorGraph.l1ID], Vector2(-37.5, -50))
    XCTAssertEqual(ATy[SimpleGaussianFactorGraph.x1ID], Vector2(-150, 25))
    XCTAssertEqual(ATy[SimpleGaussianFactorGraph.x2ID], Vector2(187.5, 25))
  }
}
