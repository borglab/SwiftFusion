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

/// A factor that selects a pose prior using an integer input index.
fileprivate struct DemoFactor<Pose: LieGroup>: GenericFactor {
  typealias Variables = Tuple2<Int, Pose>

  let edges: Variables.Indices

  let priors: [Pose]

  func error(_ variables: Variables) -> Double {
    let (priorIndex, position) = (variables.head, variables.tail.head)
    return priors[priorIndex].localCoordinate(position).squaredNorm
  }
}

class GenericFactorTests: XCTestCase {
  /// Tests errors in a simple factor graph.
  func testErrors() {
    // Set up 4 variables:
    // - `intVar1` is an `Int` with value `0`.
    // - `intVar2` is an `Int` with value `1`.
    // - `poseVar1` is a `Pose2` with value `Pose2(1, 0, 0)`.
    // - `poseVar2` is a `Pose2` with value `Pose2(0, 1, 1)`.

    let intVariables = ArrayStorage<Int>.create(minimumCapacity: 2)
    let intVar1 = TypedID<Int, Int>(intVariables.append(0)!)
    let intVar2 = TypedID<Int, Int>(intVariables.append(1)!)

    let poseVariables = ArrayStorage<Pose2>.create(minimumCapacity: 2)
    let poseVar1 = TypedID<Pose2, Int>(poseVariables.append(Pose2(1, 0, 0))!)
    let poseVar2 = TypedID<Pose2, Int>(poseVariables.append(Pose2(0, 1, 1))!)

    let variableAssignments = ValuesArray(contiguousStorage: [
      ObjectIdentifier(Int.self): intVariables,
      ObjectIdentifier(Pose2.self): poseVariables
    ])

    // Set up 4 `DemoFactor`s, connected to each combination of `Int` and `Pose2` variables.

    let priors = [Pose2(1, 0, 0), Pose2(0, 1, 1)]
    let factors = FactorArrayStorage<DemoFactor<Pose2>>.create(minimumCapacity: 4)
    _ = factors.append(DemoFactor(edges: Tuple2(intVar1, poseVar1), priors: priors))
    _ = factors.append(DemoFactor(edges: Tuple2(intVar1, poseVar2), priors: priors))
    _ = factors.append(DemoFactor(edges: Tuple2(intVar2, poseVar1), priors: priors))
    _ = factors.append(DemoFactor(edges: Tuple2(intVar2, poseVar2), priors: priors))

    // Check that the errors are what we expect.

    let errors = factors.errors(variableAssignments)
    XCTAssertEqual(errors[0], 0)
    XCTAssertEqual(errors[1], Pose2(1, 0, 0).localCoordinate(Pose2(0, 1, 1)).squaredNorm)
    XCTAssertEqual(errors[2], Pose2(1, 0, 0).localCoordinate(Pose2(0, 1, 1)).squaredNorm)
    XCTAssertEqual(errors[3], 0)
  }
}
