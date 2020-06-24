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
@testable import SwiftFusion

class GaussianFactorGraphTests: XCTestCase {
  /// Adds some scalar jacobians and tests that the resulting graph produces the correct error
  /// vectors.
  func testAddScalarJacobians() {
    var x = VariableAssignments()
    let id1 = x.store(Vector2(1, 2))
    let id2 = x.store(Vector2(3, 4))
    let id3 = x.store(Vector3(5, 6, 7))

    var jacobians = GaussianFactorGraph(storage: [:], zeroValues: x.tangentVectorZeros)
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
}
