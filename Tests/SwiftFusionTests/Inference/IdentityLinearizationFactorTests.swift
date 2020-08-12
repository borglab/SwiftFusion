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

fileprivate typealias TestGaussianFactor = JacobianFactor<Array3<Tuple1<Vector3>>, Vector3>

class IdentityLinearizationFactorTests: XCTestCase {
  /// Test that `IdentityLinearizationFactor` forwards to the underlying factor's methods.
  func testForwardsMethods() {
    let base = TestGaussianFactor(
      jacobian: Matrix3.identity,
      error: Vector3(10, 20, 30),
      edges: Tuple1(TypedID(0)))
    let f = IdentityLinearizationFactor<TestGaussianFactor>(
      linearizing: base, at: Tuple1(Vector3(0, 0, 0)))
    XCTAssertEqual(f.edges, base.edges)
    let x = Tuple1(Vector3(100, 200, 300))
    let y = Vector3(100, 200, 300)
    XCTAssertEqual(f.error(at: x), base.error(at: x))
    XCTAssertEqual(f.errorVector(at: x), base.errorVector(at: x))
    XCTAssertEqual(f.errorVector_linearComponent(x), base.errorVector_linearComponent(x))
    XCTAssertEqual(
      f.errorVector_linearComponent_adjoint(y),
      base.errorVector_linearComponent_adjoint(y))
  }
}
