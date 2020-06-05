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
fileprivate struct PriorFactor<Pose: LieGroup>: GenericFactor {
  typealias Variables = Tuple1<Pose>

  let edges: Variables.Indices

  let prior: Pose

  func error(_ variables: Variables) -> Double {
    let actualPosition = variables.head
    return prior.localCoordinate(actualPosition).squaredNorm
  }
}

/// A factor that switches between different linear motions based on an integer label.
fileprivate struct SwitchingLinearMotionFactor<Pose: LieGroup>: GenericFactor {
  typealias Variables = Tuple3<Int, Pose, Pose>

  let edges: Variables.Indices

  let motions: [Pose]

  func error(_ variables: Variables) -> Double {
    let (motionLabel, start, end) = (variables.head, variables.tail.head, variables.tail.tail.head)
    let actualMotion = between(start, end)
    return motions[motionLabel].localCoordinate(actualMotion).squaredNorm
  }
}

class GenericFactorTests: XCTestCase {
  /// Tests errors in a simple factor graph.
  func testErrors() {
    // A switching linear motion factor graph.
    //
    // There are 3 pose variables and 2 motion labels in the graph. The initial guesses are:
    // - motion labels: 0, 1
    // - poses: Pose2(0, 0, 0), Pose2(1, 1, 0), Pose2(1, 1, 0.9)
    //
    // The factors are:
    // - a prior factor giving a prior of `Pose2(0, 0, 0)` on the first position.
    // - two motion factors with possible motions [Pose2(1, 1, 0), Pose2(0, 0, 1)]

    // Set up the initial guess.

    let intVariables = ArrayStorage<Int>.create(minimumCapacity: 2)
    let motionLabel1ID = TypedID<Int, Int>(intVariables.append(0)!)
    let motionLabel2ID = TypedID<Int, Int>(intVariables.append(1)!)

    let poseVariables = ArrayStorage<Pose2>.create(minimumCapacity: 3)
    let pose1ID = TypedID<Pose2, Int>(poseVariables.append(Pose2(0, 0, 0))!)
    let pose2ID = TypedID<Pose2, Int>(poseVariables.append(Pose2(1, 1, 0))!)
    let pose3ID = TypedID<Pose2, Int>(poseVariables.append(Pose2(1, 1, 0.9))!)

    let variableAssignments = ValuesArray(contiguousStorage: [
      ObjectIdentifier(Int.self): intVariables,
      ObjectIdentifier(Pose2.self): poseVariables
    ])

    // Set up the factor graph.

    let priorFactors = FactorArrayStorage<PriorFactor<Pose2>>.create(minimumCapacity: 1)
    _ = priorFactors.append(PriorFactor(edges: Tuple1(pose1ID), prior: Pose2(0, 0, 0)))

    let motionFactors =
      FactorArrayStorage<SwitchingLinearMotionFactor<Pose2>>.create(minimumCapacity: 2)
    _ = motionFactors.append(SwitchingLinearMotionFactor(
      edges: Tuple3(motionLabel1ID, pose1ID, pose2ID),
      motions: [Pose2(1, 1, 0), Pose2(0, 0, 1)]
    ))
    _ = motionFactors.append(SwitchingLinearMotionFactor(
      edges: Tuple3(motionLabel2ID, pose2ID, pose3ID),
      motions: [Pose2(1, 1, 0), Pose2(0, 0, 1)]
    ))

    // Check that the errors are what we expect.

    let priorErrors = priorFactors.errors(variableAssignments)
    XCTAssertEqual(priorErrors[0], 0)

    let motionErrors = motionFactors.errors(variableAssignments)
    XCTAssertEqual(motionErrors[0], 0)
    XCTAssertEqual(motionErrors[1], Vector3(0.1, 0, 0).squaredNorm, accuracy: 1e-6)
  }
}
