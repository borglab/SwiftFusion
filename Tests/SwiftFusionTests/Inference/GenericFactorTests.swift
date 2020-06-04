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

/// A factor that specifies a prior on a pose.
///
/// Note: This is currently named with a "Generic" prefix to avoid clashing with the other factors.
/// When we completely replace the existing factors with the "Generic" ones, we should remove this
/// prefix.
fileprivate struct GenericPriorFactor<Pose: LieGroup>: GenericLinearizableFactor {
  typealias Variables = Tuple1<Pose>

  let edges: Variables.Indices
  let prior: Pose

  typealias ErrorVector = Pose.TangentVector
  func errorVector(_ x: Pose) -> ErrorVector {
    return prior.localCoordinate(x)
  }

  // Note: All the remaining code in this factor is boilerplate that we can eventually eliminate
  // with sugar.

  func error(_ variables: Variables) -> Double {
    return errorVector(variables).squaredNorm
  }

  func errorVector(_ variables: Variables) -> Pose.TangentVector {
    return errorVector(variables.head)
  }

  typealias Linearized = GenericJacobianFactor<Variables.TangentVector, ErrorVector>
  func linearized(_ variables: Variables) -> Linearized {
    Linearized(linearizing: errorVector, at: variables.head, edges: edges)
  }
}

/// A factor that switches between different linear motions based on an integer label.
fileprivate struct SwitchingMotionModelFactor<Pose: LieGroup>: GenericLinearizableFactor {
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

  func error(_ variables: Variables) -> Double {
    return errorVector(variables).squaredNorm
  }

  func errorVector(_ variables: Variables) -> Pose.TangentVector {
    return errorVector(variables.head, variables.tail.head, variables.tail.tail.head)
  }

  typealias Linearized = GenericJacobianFactor<Variables.Tail.TangentVector, ErrorVector>
  func linearized(_ variables: Variables) -> Linearized {
    fatalError()
  }
}

/// A Gaussian factor stored as a jacobian matrix and an error vector.
///
/// Note: This is currently named with a "Generic" prefix to avoid clashing with the other factors.
/// When we completely replace the existing factors with the "Generic" ones, we should remove this
/// prefix.
fileprivate struct GenericJacobianFactor<
  Variables: EuclideanVectorN & VariableTuple,
  ErrorVector: EuclideanVectorN
>: GenericGaussianFactor {
  /// The Jacobian matrix.
  ///
  /// Note: It's inefficient to use this dynamically-shaped matrix type, so we want to eventually
  /// switch this to a fixed-size matrix type.
  let jacobian: Matrix

  /// The error vector.
  let error: ErrorVector

  /// The ids of the variables adjacent to this factor.
  let edges: Variables.Indices

  /// Creates a Jacobian factor with the given `jacobian`, `error`, and `edges`.
  init(jacobian: Matrix, error: ErrorVector, edges: Variables.Indices) {
    self.jacobian = jacobian
    self.error = error
    self.edges = edges
  }

  /// Creates a Jacobian factor that linearizes `f` at `x`, and is adjacent to the variables
  /// identifed by edges.
  init<Input: Differentiable>(
    linearizing f: @differentiable (Input) -> ErrorVector,
    at x: Input,
    edges: Tuple<TypedID<Input, Int>, Empty>
  ) where Variables == Tuple<Input.TangentVector, Empty> {
    let (error, pb) = valueWithPullback(at: x, in: f)
    self.error = error
    self.jacobian = Matrix(stacking: ErrorVector.standardBasis.map { basisVector in
      return pb(basisVector).vector
    })

    // TODO: Note that this is wrong when there are different types of input variables that have
    // the same tangent vector.
    self.edges = Tuple1(TypedID(edges.head.perTypeID))
  }

  func error(_ variables: Variables) -> Double {
    return errorVector(variables).squaredNorm
  }

  func errorVector(_ variables: Variables) -> ErrorVector {
    return linearForward(variables) + error
  }

  func linearForward(_ variables: Variables) -> ErrorVector {
    return ErrorVector(matvec(jacobian, variables.vector))
  }

  func linearAdjoint(_ errorVector: ErrorVector) -> Variables {
    return Variables(matvec(jacobian, transposed: true, errorVector.vector))
  }

  typealias Linearized = Self
  func linearized(_: Variables) -> Self {
    return self
  }
}

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
  var initialGuess = ValuesArray(contiguousStorage: [:])
  var motionLabelIDs = [TypedID<Int, Int>]()
  var poseIDs = [TypedID<Pose2, Int>]()

  var priorFactors = ArrayBuffer<FactorArrayStorage<GenericPriorFactor<Pose2>>>()
  var motionFactors = ArrayBuffer<FactorArrayStorage<SwitchingMotionModelFactor<Pose2>>>()

  init() {
    // Set up the initial guess.

    var motionLabelVariables = ArrayBuffer<ArrayStorage<Int>>()
    motionLabelIDs.append(TypedID(motionLabelVariables.append(0)))
    motionLabelIDs.append(TypedID(motionLabelVariables.append(1)))

    var poseVariables = ArrayBuffer<ArrayStorage<Pose2>>()
    poseIDs.append(TypedID(poseVariables.append(Pose2(0, 0, 0))))
    poseIDs.append(TypedID(poseVariables.append(Pose2(1, 1, 0))))
    poseIDs.append(TypedID(poseVariables.append(Pose2(1, 1, 0.9))))

    initialGuess = ValuesArray(contiguousStorage: [
      ObjectIdentifier(Int.self): AnyArrayBuffer(motionLabelVariables),
      ObjectIdentifier(Pose2.self): AnyArrayBuffer(poseVariables)
    ])

    // Set up the factor graph.

    _ = priorFactors.append(GenericPriorFactor(edges: Tuple1(poseIDs[0]), prior: Pose2(0, 0, 0)))

    _ = motionFactors.append(SwitchingMotionModelFactor(
      edges: Tuple3(motionLabelIDs[0], poseIDs[0], poseIDs[1]),
      motions: [Pose2(1, 1, 0), Pose2(0, 0, 1)]
    ))
    _ = motionFactors.append(SwitchingMotionModelFactor(
      edges: Tuple3(motionLabelIDs[1], poseIDs[1], poseIDs[2]),
      motions: [Pose2(1, 1, 0), Pose2(0, 0, 1)]
    ))
  }
}

class GenericFactorTests: XCTestCase {
  /// Tests errors in the example factor graph.
  func testErrors() {
    let graph = ExampleFactorGraph()

    let priorErrors = graph.priorFactors.errors(graph.initialGuess)
    XCTAssertEqual(priorErrors[0], 0)

    let motionErrors = graph.motionFactors.errors(graph.initialGuess)
    XCTAssertEqual(motionErrors[0], 0)
    XCTAssertEqual(motionErrors[1], Vector3(0.1, 0, 0).squaredNorm, accuracy: 1e-6)
  }

  /// Test linearizing the example factor graph.
  func testLinearized() {
    let graph = ExampleFactorGraph()

    let priorsLinearized = graph.priorFactors.linearized(graph.initialGuess)
    XCTAssertEqual(priorsLinearized[0].jacobian, Matrix(eye: 3))
    XCTAssertEqual(priorsLinearized[0].error, Vector3(0, 0, 0))
  }

  /// Test the `errorVectors`, `linearForward`, and `linearAdjoint` operations on a collection of
  /// Gaussian factors.
  func testGaussianFactorOperations() {
    var vectorVariables = ArrayBuffer<ArrayStorage<Vector2>>()
    let vectorVar1ID = TypedID<Vector2, Int>(vectorVariables.append(Vector2(1, 2)))
    let variableAssignments = ValuesArray(contiguousStorage: [
      ObjectIdentifier(Vector2.self): AnyArrayBuffer(vectorVariables)
    ])

    var factors =
      ArrayBuffer<GaussianFactorArrayStorage<GenericJacobianFactor<Tuple1<Vector2>, Vector2>>>()
    let matrix1 = Matrix(scalars: [1, 1, 0, 1], rowCount: 2, columnCount: 2)
    _ = factors.append(GenericJacobianFactor(
      jacobian: matrix1,
      error: Vector2(0, 0),
      edges: Tuple1(vectorVar1ID)
    ))
    let matrix2 = Matrix(scalars: [0, 3, 7, 0], rowCount: 2, columnCount: 2)
    _ = factors.append(GenericJacobianFactor(
      jacobian: matrix2,
      error: Vector2(100, 200),
      edges: Tuple1(vectorVar1ID)
    ))

    let errorVectors = factors.errorVectors(variableAssignments)
    XCTAssertEqual(errorVectors[0], Vector2(3, 2))  // matrix1 * [1 2] + [0 0]
    XCTAssertEqual(errorVectors[1], Vector2(106, 207))  // matrix2 * [1 2] + [100 200]

    let forwardResult = factors.linearForward(variableAssignments)
    XCTAssertEqual(forwardResult[0], Vector2(3, 2))  // matrix1 * [1 2]
    XCTAssertEqual(forwardResult[1], Vector2(6, 7))  // matrix2 * [1 2]

    var adjointResult = ValuesArray(contiguousStorage: [
      ObjectIdentifier(Vector2.self): AnyArrayBuffer(
        ArrayBuffer<VectorArrayStorage<Vector2>>([Vector2(0, 0)])
      )
    ])
    forwardResult.withUnsafeBufferPointer { b in
      factors.linearAdjoint(b.baseAddress!, into: &adjointResult)
    }

    // matrix1^T * [3 2] + matrix2^T * [6 7]
    adjointResult.contiguousStorage[ObjectIdentifier(Vector2.self)]!
      .withUnsafeRawPointerToElements { p in
        XCTAssertEqual(p.assumingMemoryBound(to: Vector2.self).pointee, Vector2(52, 23))
      }
  }
}
