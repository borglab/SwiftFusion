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

class PPCATests: XCTestCase {
  func testSimplePPCAForwardZero() {
    var x = VariableAssignments()
    let a = x.store(Vector5(1, 1, 1, 1, 1))
    let imagePatch = Tensor10x10(Tensor<Double>(zeros: [10, 10]))
    let mu = Tensor10x10(Tensor<Double>(zeros: [10, 10]))
    let W: Tensor<Double> = Tensor<Double>(zeros: [10, 10, 5])
    var graph = FactorGraph()
    graph.store(PPCAFactor(a, desired: imagePatch, W: W, mu: mu))
    graph.store(PriorFactor<Vector5>(a, Vector5(1, 1, 1, 1, 1)))

    XCTAssertEqual(graph.error(at: x), 0.0)
  }

  func testSimplePPCAForwardSimple() {
    var x = VariableAssignments()
    let a = x.store(Vector5(1, 1, 1, 1, 1))
    let imagePatch = Tensor10x10(Tensor<Double>(zeros: [10, 10]))
    let mu = Tensor10x10(Tensor<Double>(zeros: [10, 10]))
    var W: Tensor<Double> = Tensor<Double>(zeros: [10, 10, 5])
    W[0, 0, 0] = Tensor(1.0)
    var graph = FactorGraph()
    graph.store(PPCAFactor(a, desired: imagePatch, W: W, mu: mu))
    graph.store(PriorFactor<Vector5>(a, Vector5(1, 1, 1, 1, 1)))

    XCTAssertEqual(graph.error(at: x), 0.5)
  }

  func testSimplePPCALinearization() {
    let imagePatch = Tensor10x10(Tensor<Double>(zeros: [10, 10]))
    let mu = Tensor10x10(Tensor<Double>(zeros: [10, 10]))

    var W: Tensor<Double> = Tensor<Double>(zeros: [10, 10, 5])
    W[0, 0, 0] = Tensor(1.0)
    W[0, 0, 1] = Tensor(1.0)
    W[0, 0, 2] = Tensor(1.0)
    W[0, 0, 3] = Tensor(1.0)
    W[0, 0, 4] = Tensor(1.0)

    let ppcaFactor = PPCAFactor(TypedID(0), desired: imagePatch, W: W, mu: mu)
    let jf = JacobianFactor100x5_1(linearizing: ppcaFactor, at: Tuple1(Vector5(1, 1, 1, 1, 1)))
    let jacobianPPCA = Tensor<Double>(stacking: jf.jacobian.map { $0.flatTensor })

    var expectedJacobianPPCA = Tensor<Double>(zeros: [100, 5])
    expectedJacobianPPCA[0, 0] = Tensor(1.0)
    expectedJacobianPPCA[0, 1] = Tensor(1.0)
    expectedJacobianPPCA[0, 2] = Tensor(1.0)
    expectedJacobianPPCA[0, 3] = Tensor(1.0)
    expectedJacobianPPCA[0, 4] = Tensor(1.0)

    assertEqual(jacobianPPCA, expectedJacobianPPCA, accuracy: 1e-10)
  }

  func testSimplePPCASolve() {
    var x = VariableAssignments()
    let a = x.store(Vector5(1, 1, 1, 1, 1))

    /// imagePatch will only have one pixel with value 1
    var imagePatch = Tensor10x10(Tensor<Double>(zeros: [10, 10]))
    imagePatch.tensor[0, 0] = Tensor(1.0)

    let mu = Tensor10x10(Tensor<Double>(zeros: [10, 10]))

    /// Weight only applies to first pixel
    var W: Tensor<Double> = Tensor<Double>(zeros: [10, 10, 5])
    W[0, 0, 0] = Tensor(1.0)
    W[0, 0, 1] = Tensor(-1.0)
    W[0, 0, 2] = Tensor(0.0)
    W[0, 0, 3] = Tensor(1.0)
    W[0, 0, 4] = Tensor(-1.0)

    var graph = FactorGraph()
    graph.store(PPCAFactor(a, desired: imagePatch, W: W, mu: mu))
    graph.store(PriorFactor<Vector5>(a, Vector5(1, 1, 1, 1, 1)))

    for _ in 0..<2 {
      let gfg = graph.linearized(at: x)
      var dx = x.tangentVectorZeros
      var optimizer = GenericCGLS(precision: 1e-6, max_iteration: 500)
      optimizer.optimize(gfg: gfg, initial: &dx)
      x.move(along: dx)
    }

    assertAllKeyPathEqual(x[a], Vector5(1.2, 0.8, 1.0, 1.2, 0.8), accuracy: 1e-7)
  }
}
