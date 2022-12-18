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

import _Differentiation
import Foundation
// import TensorFlow
import XCTest

import PenguinStructures
@testable import SwiftFusion

class PPCATests: XCTestCase {
  /// Tests that the derivative of the new AppearanceTrackingFactor matches the old PPCATrackingFactor
  func testLinearizedValue() {
    let factor = PPCATrackingFactor.testFixture(TypedID<Pose2>(0), TypedID<Vector5>(0), seed: (4, 4))

    let ppca = PPCA(W: factor.W, mu: factor.mu.tensor)
    let generic_factor = AppearanceTrackingFactor(
      TypedID<Pose2>(0), TypedID<Vector5>(0),
      measurement: Tensor<Float>(factor.measurement),
      appearanceModel: ppca.decode, appearanceModelJacobian: { _ in ppca.W }
    )

    for _ in 0..<2 {
      let linearizationPoint = Tuple2(
        Pose2(randomWithCovariance: eye(rowCount: 3), seed: (5, 5)),
        Vector5(flatTensor: Tensor(randomNormal: [5], seed: (6, 6))))

      typealias Variables = PPCATrackingFactor.Variables.TangentVector
      typealias ErrorVector = PPCATrackingFactor.ErrorVector

      // Below we compare custom and much faster differentiation with the autodiff version.

      let autodiff = JacobianFactor<Array<Variables>, ErrorVector>(
        linearizing: factor, at: linearizationPoint)
      let custom = generic_factor.linearized(at: linearizationPoint)

      // Compare the linearizations at zero (the error vector).
      XCTAssertEqual(
        custom.errorVector(at: Variables.zero), autodiff.errorVector(at: Variables.zero))

      // Compare the Jacobian-vector-products (forward derivative).
      for _ in 0..<10 {
        let v = Variables(flatTensor: Tensor(randomNormal: [Variables.dimension], seed: (7, 7)))
        assertEqual(
          custom.errorVector_linearComponent(v).tensor,
          autodiff.errorVector_linearComponent(v).tensor,
          accuracy: 1e-4)
      }

      // Compare the vector-Jacobian-products (reverse derivative).
      for _ in 0..<10 {
        let e = ErrorVector(Tensor(randomNormal: factor.mu.shape, seed: (8, 8)))
        assertEqual(
          custom.errorVector_linearComponent_adjoint(e).flatTensor,
          autodiff.errorVector_linearComponent_adjoint(e).flatTensor,
          accuracy: 1e-4)
      }
    }
  }

  /// Test that the VJP of the AppearanceTrackingFactor is the same as the automatically derived VJP of the PPCATrackingFactor.
  func testVJP() {
    let factor = PPCATrackingFactor.testFixture(TypedID<Pose2>(0), TypedID<Vector5>(0), seed: (4, 4))

    let ppca = PPCA(W: factor.W, mu: factor.mu.tensor)
    let generic_factor = AppearanceTrackingFactor(
      TypedID<Pose2>(0), TypedID<Vector5>(0),
      measurement: Tensor<Float>(factor.measurement),
      appearanceModel: ppca.decode, appearanceModelJacobian: { _ in ppca.W }
    )

    for _ in 0..<5 {
      let linearizationPoint = Tuple2(
        Pose2(randomWithCovariance: eye(rowCount: 3), seed: (5, 5)),
        Vector5(flatTensor: Tensor(randomNormal: [5], seed: (6, 6))))

      let pbFactor = pullback(at: linearizationPoint) { factor.errorVector(at: $0) }
      let pbGeneric_factor = pullback(at: linearizationPoint) { generic_factor.errorVector(at: $0) }

      let tangentVector = TensorVector(Tensor<Double>(randomNormal: factor.mu.tensor.shape))

      assertEqual(
        pbFactor(tangentVector).flatTensor,
        pbGeneric_factor(tangentVector).flatTensor,
        accuracy: 1e-4)
    }
  }

  /// Tests that factor graphs with a `PPCATrackingFactor`s linearize them using the custom
  /// linearization.
  func testFactorGraphUsesCustomLinearized() {
    var x = VariableAssignments()
    let poseId = x.store(Pose2(100, 100, 0))
    let latentId = x.store(Vector5.zero)

    var fg = FactorGraph()
    fg.store(PPCATrackingFactor.testFixture(poseId, latentId, seed: (9, 9)))

    let gfg = fg.linearized(at: x)

    // Assert that the linearized graph has the custom `LinearizedPPCATrackingFactor` that uses our
    // faster custom Jacobian calculate the default `JacobianFactor` that calculates the Jacobian
    // using autodiff.
    //
    // This works by checking that the first (only) element of the graph's storage can be cast to
    // an array of `LinearizedPPCATrackingFactor`.
    XCTAssertNotNil(gfg.storage.first!.value[elementType: Type<LinearizedPPCATrackingFactor>()])
  }

  /// Tests that factor graphs with a `PPCATrackingFactor`s linearize them using the custom
  /// linearization.
  func testFactorGraphUsesCustomLinearizedGenerativeFactor() {
    let ppca_factor = PPCATrackingFactor.testFixture(TypedID<Pose2>(0), TypedID<Vector5>(0), seed: (4, 4))
    let ppca = PPCA(W: ppca_factor.W.tiled(multiples: [1, 1, 3, 1]), mu: ppca_factor.mu.tensor.tiled(multiples: [1, 1, 3]))
    let generic_factor = AppearanceTrackingFactor(
      TypedID<Pose2>(0), TypedID<Vector5>(0),
      measurement: Tensor<Float>(ppca_factor.measurement.tiled(multiples: [1, 1, 3])),
      appearanceModel: ppca.decode, appearanceModelJacobian: { _ in ppca.W }
    )

    var x = VariableAssignments()
    let _ = x.store(Pose2(100, 100, 0))
    let _ = x.store(Vector5.zero)

    var fg = FactorGraph()
    fg.store(generic_factor)

    let gfg = fg.linearized(at: x)

    // Assert that the linearized graph has the custom `LinearizedAppearanceTrackingFactor` that uses our
    // faster custom Jacobian calculate the default `JacobianFactor` that calculates the Jacobian
    // using autodiff.
    //
    // This works by checking that the first (only) element of the graph's storage can be cast to
    // an array of `LinearizedAppearanceTrackingFactor<Vector5>`.
    XCTAssertNotNil(
      gfg.storage.first!.value[elementType: Type<LinearizedAppearanceTrackingFactor<Vector5>>()])
  }

}
