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

class PPCATrackingFactorTests: XCTestCase {
  /// Tests that the custom `linearized(at:)` method produces the same results at autodiff.
  func testLinearizedValue() {
    // Initialize `factor` with random fields.
    let image = Tensor<Double>(randomNormal: [500, 500, 1])
    let W = Tensor<Double>(
      randomNormal: PPCATrackingFactor.Patch.shape + [PPCATrackingFactor.V1.dimension])
    let mu = PPCATrackingFactor.Patch(
      Tensor<Double>(randomNormal: PPCATrackingFactor.Patch.shape))
    let poseID = TypedID<Pose2>(0)
    let latentID = TypedID<Vector5>(0)
    let factor = PPCATrackingFactor(poseID, latentID, measurement: image, W: W, mu: mu)

    for _ in 0..<2 {
      let linearizationPoint = Tuple2(
        Pose2(randomWithCovariance: eye(rowCount: 3)),
        Vector5(flatTensor: Tensor(randomNormal: [5])))

      typealias Variables = PPCATrackingFactor.Variables.TangentVector
      typealias ErrorVector = PPCATrackingFactor.ErrorVector

      let autodiff = JacobianFactor<Array<Variables>, ErrorVector>(
        linearizing: factor, at: linearizationPoint)
      let custom = factor.linearized(at: linearizationPoint)

      XCTAssertEqual(
        custom.errorVector(at: Variables.zero), autodiff.errorVector(at: Variables.zero))

      for _ in 0..<10 {
        let v = Variables(flatTensor: Tensor(randomNormal: [Variables.dimension]))
        assertEqual(
          custom.errorVector_linearComponent(v).tensor,
          autodiff.errorVector_linearComponent(v).tensor,
          accuracy: 1e-6)
      }

      for _ in 0..<10 {
        let e = ErrorVector(Tensor(randomNormal: ErrorVector.shape))
        assertEqual(
          custom.errorVector_linearComponent_adjoint(e).flatTensor,
          autodiff.errorVector_linearComponent_adjoint(e).flatTensor,
          accuracy: 1e-6)
      }
    }
  }

  /// Tests that factor graphs with a `PPCATrackingFactor`s linearize them using the custom
  /// linearization.
  func testFactorGraphUsesCustomLinearized() {
    let image = Tensor<Double>(randomNormal: [500, 500, 1])
    let W = Tensor<Double>(
      randomNormal: PPCATrackingFactor.Patch.shape + [PPCATrackingFactor.V1.dimension])
    let mu = PPCATrackingFactor.Patch(
      Tensor<Double>(randomNormal: PPCATrackingFactor.Patch.shape))

    var x = VariableAssignments()
    let poseId = x.store(Pose2(100, 100, 0))
    let latentId = x.store(Vector5.zero)

    var fg = FactorGraph()
    fg.store(PPCATrackingFactor(poseId, latentId, measurement: image, W: W, mu: mu))

    let gfg = fg.linearized(at: x)
    XCTAssertNotNil(gfg.storage.first!.value[elementType: Type<LinearizedPPCATrackingFactor>()])
  }
}
