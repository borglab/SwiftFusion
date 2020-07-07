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

/// A factor that switches between different linear motions based on an integer label.
fileprivate struct SwitchingMotionModelFactor<Pose: LieGroup, JacobianRows: FixedSizeArray>:
  LinearizableFactor
  where JacobianRows.Element == Tuple2<Pose.TangentVector, Pose.TangentVector>
{
  typealias Variables = Tuple3<Int, Pose, Pose>

  let edges: Variables.Indices

  let motions: [Pose]

  typealias ErrorVector = Pose.TangentVector
  func errorVector(_ motionLabel: Int, _ start: Pose, _ end: Pose) -> ErrorVector {
    let actualMotion = between(start, end)
    return motions[motionLabel].localCoordinate(actualMotion)
  }

  // Note: All the remaining code in this factor is boilerplate that we can eventually eliminate
  // with sugar.

  func error(at x: Variables) -> Double {
    return 0.5 * errorVector(at: x).squaredNorm
  }

  func errorVector(at x: Variables) -> Pose.TangentVector {
    return errorVector(x.head, x.tail.head, x.tail.tail.head)
  }

  typealias Linearization = JacobianFactor<JacobianRows, ErrorVector>
  func linearized(at x: Variables) -> Linearization {
    Linearization(
      linearizing: { errorVector(x.head, $0.head, $0.tail.head) },
      at: x.tail,
      edges: edges.tail
    )
  }
}

fileprivate typealias SwitchingMotionModelFactor2 =
  SwitchingMotionModelFactor<Pose2, Array3<Tuple2<Vector3, Vector3>>>

// A switching motion model factor graph.
//
// There are 3 pose variables and 2 motion labels in the graph. The initial guesses are:
// - motion labels: 0, 1
// - poses: Pose2(0, 0, 0), Pose2(1, 1, 0), Pose2(1, 1, 0.9)
//
// The factors are:
// - a prior factor giving a prior of `Pose2(0, 0, 0)` on the first position.
// - two motion factors with possible motions [Pose2(1, 1, 0), Pose2(0, 0, 1)]
fileprivate struct ExampleFactorGraph {
  var initialGuess = VariableAssignments()
  var motionLabelIDs = [TypedID<Int, Int>]()
  var poseIDs = [TypedID<Pose2, Int>]()

  var priorFactors = AnyLinearizableFactorArrayBuffer(ArrayBuffer<PriorFactor2>())
  var motionFactors = AnyLinearizableFactorArrayBuffer(ArrayBuffer<SwitchingMotionModelFactor2>())

  init() {
    // Set up the initial guess.

    motionLabelIDs.append(initialGuess.store(0))
    motionLabelIDs.append(initialGuess.store(1))

    poseIDs.append(initialGuess.store(Pose2(0, 0, 0)))
    poseIDs.append(initialGuess.store(Pose2(1, 1, 0)))
    poseIDs.append(initialGuess.store(Pose2(1, 1, 0.9)))

    // Set up the factor graph.

    _ = priorFactors.unsafelyAppend(PriorFactor2(poseIDs[0], Pose2(0, 0, 0)))

    _ = motionFactors.unsafelyAppend(SwitchingMotionModelFactor2(
      edges: Tuple3(motionLabelIDs[0], poseIDs[0], poseIDs[1]),
      motions: [Pose2(1, 1, 0), Pose2(0, 0, 1)]
    ))
    _ = motionFactors.unsafelyAppend(SwitchingMotionModelFactor2(
      edges: Tuple3(motionLabelIDs[1], poseIDs[1], poseIDs[2]),
      motions: [Pose2(1, 1, 0), Pose2(0, 0, 1)]
    ))
  }
}

class FactorTests: XCTestCase {
  /// Tests errors in the example factor graph.
  func testErrors() {
    let graph = ExampleFactorGraph()

    let priorErrors = graph.priorFactors.errors(at: graph.initialGuess)
    XCTAssertEqual(priorErrors[0], 0)

    let motionErrors = graph.motionFactors.errors(at: graph.initialGuess)
    XCTAssertEqual(motionErrors[0], 0)
    XCTAssertEqual(motionErrors[1], 0.5 * Vector3(0.1, 0, 0).squaredNorm, accuracy: 1e-6)
  }

  /// Test the error vectors from the example factor graph.
  func testErrorVectors() {
    let graph = ExampleFactorGraph()

    let priorErrorVectors = ArrayBuffer<Vector3>(
      unsafelyDowncasting: graph.priorFactors.errorVectors(at: graph.initialGuess))
    XCTAssertEqual(priorErrorVectors[0], Vector3(0, 0, 0))

    let motionErrorVectors = ArrayBuffer<Vector3>(
      unsafelyDowncasting: graph.motionFactors.errorVectors(at: graph.initialGuess))
    XCTAssertEqual(motionErrorVectors[0], Vector3(0, 0, 0))
    assertAllKeyPathEqual(motionErrorVectors[1], Vector3(-0.1, 0, 0), accuracy: 1e-6)
  }

  /// Test linearizing the example factor graph.
  func testLinearized() {
    let graph = ExampleFactorGraph()

    let priorsLinearized = ArrayBuffer<JacobianFactor3x3_1>(
      unsafelyDowncasting: graph.priorFactors.linearized(at: graph.initialGuess))
    XCTAssertEqual(priorsLinearized[0].jacobian, Array3(
      Tuple1(Vector3(1, 0, 0)),
      Tuple1(Vector3(0, 1, 0)),
      Tuple1(Vector3(0, 0, 1))
    ))
    XCTAssertEqual(priorsLinearized[0].error, Vector3(0, 0, 0))

    let motionsLinearized = ArrayBuffer<JacobianFactor3x3_2>(
      unsafelyDowncasting: graph.motionFactors.linearized(at: graph.initialGuess))
    XCTAssertEqual(motionsLinearized[0].jacobian, Array3(
      Tuple2(Vector3(-1, 0, 0), Vector3(1, 0, 0)),
      Tuple2(Vector3(1, -1, 0), Vector3(0, 1, 0)),
      Tuple2(Vector3(-1, 0, -1), Vector3(0, 0, 1))
    ))
    XCTAssertEqual(motionsLinearized[0].error, Vector3(0, 0, 0))
    assertAllKeyPathEqual(
      motionsLinearized[1].jacobian.map { $0.head },
      [
        Vector3(-1.0, 0.0, 0.0),
        Vector3(0.0, -0.62160997, -0.78332691),
        Vector3(0.0, 0.78332691, -0.62160997)
      ],
      accuracy: 1e-6
    )
    assertAllKeyPathEqual(
      motionsLinearized[1].jacobian.map { $0.tail.head },
      [
        Vector3(1, 0, 0),
        Vector3(0, 1, 0),
        Vector3(0, 0, 1)
      ],
      accuracy: 1e-6
    )
    assertAllKeyPathEqual(motionsLinearized[1].error, Vector3(0.1, 0, 0), accuracy: 1e-6)
  }

  /// Test the `errorVectors`, `errorVector_linearComponent`, and `errorVector_linearComponent_adjoint` operations on a collection of
  /// Gaussian factors.
  func testGaussianFactorOperations() {
    var variableAssignments = VariableAssignments()
    let vectorVar1ID = variableAssignments.store(Vector2(1, 2))

    var factors = AnyGaussianFactorArrayBuffer(
      ArrayBuffer<JacobianFactor<Array2<Tuple1<Vector2>>, Vector2>>())
    let matrix1 = Array2(
      Tuple1(Vector2(1, 1)),
      Tuple1(Vector2(0, 1))
    )
    _ = factors.unsafelyAppend(JacobianFactor(
      jacobian: matrix1,
      error: Vector2(0, 0),
      edges: Tuple1(vectorVar1ID)
    ))
    let matrix2 = Array2(
      Tuple1(Vector2(0, 3)),
      Tuple1(Vector2(7, 0))
    )
    _ = factors.unsafelyAppend(JacobianFactor(
      jacobian: matrix2,
      error: Vector2(100, 200),
      edges: Tuple1(vectorVar1ID)
    ))

    let errorVectors = ArrayBuffer<Vector2>(
      unsafelyDowncasting: factors.errorVectors(at: variableAssignments))
    XCTAssertEqual(errorVectors[0], Vector2(3, 2))  // matrix1 * [1 2] - zero
    XCTAssertEqual(errorVectors[1], Vector2(-94, -193))  // matrix2 * [1 2] - [100 200]
    let forwardResult = ArrayBuffer<Vector2>(
      unsafelyDowncasting: factors.errorVectors_linearComponent(variableAssignments))
    XCTAssertEqual(forwardResult[0], Vector2(3, 2))  // matrix1 * [1 2]
    XCTAssertEqual(forwardResult[1], Vector2(6, 7))  // matrix2 * [1 2]

    var adjointResult = VariableAssignments()
    let adjointResultId = adjointResult.store(Vector2(0, 0))
    factors.errorVectors_linearComponent_adjoint(
      AnyElementArrayBuffer(forwardResult), into: &adjointResult)

    // matrix1^T * [3 2] + matrix2^T * [6 7]
    XCTAssertEqual(adjointResult[adjointResultId], Vector2(52, 23))
  }
}
