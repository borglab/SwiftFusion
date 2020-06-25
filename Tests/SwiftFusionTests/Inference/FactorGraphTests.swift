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

class FactorGraphTests: XCTestCase {
  /// Tests that we can read factors from a factor graph.
  func testGetFactors() {
    let pose1ID = TypedID<Pose2, Int>(0)
    let pose2ID = TypedID<Pose2, Int>(1)

    var graph = FactorGraph()
    graph.store(PriorFactor2(pose1ID, Pose2(1, 2, 3)))
    graph.store(PriorFactor2(pose2ID, Pose2(4, 5, 6)))
    graph.store(BetweenFactor2(pose1ID, pose2ID, Pose2(7, 8, 9)))

    XCTAssertEqual(
      graph.factors(type: PriorFactor2.self).map { $0.edges },
      [Tuple1(pose1ID), Tuple1(pose2ID)]
    )
    XCTAssertEqual(
      graph.factors(type: PriorFactor2.self).map { $0.prior },
      [Pose2(1, 2, 3), Pose2(4, 5, 6)]
    )

    XCTAssertEqual(
      graph.factors(type: BetweenFactor2.self).map { $0.edges },
      [Tuple2(pose1ID, pose2ID)]
    )
    XCTAssertEqual(
      graph.factors(type: BetweenFactor2.self).map { $0.difference },
      [Pose2(7, 8, 9)]
    )

    XCTAssertEqual(graph.factors(type: PriorFactor3.self).count, 0)
  }

  func testSimplePose2SLAM() {
    var x = VariableAssignments()
    let pose1ID = x.store(Pose2(Rot2(0.2), Vector2(0.5, 0.0)))
    let pose2ID = x.store(Pose2(Rot2(-0.2), Vector2(2.3, 0.1)))
    let pose3ID = x.store(Pose2(Rot2(.pi / 2), Vector2(4.1, 0.1)))
    let pose4ID = x.store(Pose2(Rot2(.pi), Vector2(4.0, 2.0)))
    let pose5ID = x.store(Pose2(Rot2(-.pi / 2), Vector2(2.1, 2.1)))

    var graph = FactorGraph()
    graph.store(BetweenFactor2(pose2ID, pose1ID, Pose2(2.0, 0.0, .pi / 2)))
    graph.store(BetweenFactor2(pose3ID, pose2ID, Pose2(2.0, 0.0, .pi / 2)))
    graph.store(BetweenFactor2(pose4ID, pose3ID, Pose2(2.0, 0.0, .pi / 2)))
    graph.store(BetweenFactor2(pose5ID, pose4ID, Pose2(2.0, 0.0, .pi / 2)))
    graph.store(PriorFactor2(pose1ID, Pose2(0, 0, 0)))

    for _ in 0..<3 {
      let linearized = graph.linearized(at: x)
      var dx = x.tangentVectorZeros
      var optimizer = GenericCGLS(precision: 1e-6, max_iteration: 500)
      optimizer.optimize(gfg: linearized, initial: &dx)
      x.move(along: (-1) * dx)
    }

    // Test condition: pose 5 should be identical to pose 1 (close loop).
    XCTAssertEqual(between(x[pose1ID], x[pose5ID]).t.norm, 0.0, accuracy: 1e-2)
  }
  
  /// circlePose3 generates a set of poses in a circle. This function
  /// returns those poses inside a gtsam.Values object, with sequential
  /// keys starting from 0. An optional character may be provided, which
  /// will be stored in the msb of each key (i.e. gtsam.Symbol).

  /// We use aerospace/navlab convention, X forward, Y right, Z down
  /// First pose will be at (R,0,0)
  /// ^y   ^ X
  /// |    |
  /// z-->xZ--> Y  (z pointing towards viewer, Z pointing away from viewer)
  /// Vehicle at p0 is looking towards y axis (X-axis points towards world y)
  func circlePose3(numPoses: Int = 8, radius: Double = 1.0) -> (Array<TypedID<Pose3, Int>>, VariableAssignments) {
    var ids: Array<TypedID<Pose3, Int>> = []
    var values = VariableAssignments()
    var theta = 0.0
    let dtheta = 2.0 * .pi / Double(numPoses)
    let gRo = Rot3(0, 1, 0, 1, 0, 0, 0, 0, -1)
    for _ in 0..<numPoses {
      let gti = Vector3(radius * cos(theta), radius * sin(theta), 0)
      let oRi = Rot3.fromTangent(Vector3(0, 0, -theta))  // negative yaw goes counterclockwise, with Z down !
      let gTi = Pose3(gRo * oRi, gti)
      let id = values.store(gTi)
      ids.append(id)
      theta = theta + dtheta
    }
    return (ids, values)
  }
  
  func testGtsamPose3SLAMExample() {
    // Create a hexagon of poses
    let (hexagonId, hexagon) = circlePose3(numPoses: 6, radius: 1.0)
    let p0 = hexagon[hexagonId[0]]
    let p1 = hexagon[hexagonId[1]]
    
    var x = VariableAssignments()
    
    let s = 0.10
    let id0 = x.store(p0)
    let id1 = x.store(hexagon[hexagonId[1]].retract(Vector6(s * Tensor<Double>(randomNormal: [6]))))
    let id2 = x.store(hexagon[hexagonId[2]].retract(Vector6(s * Tensor<Double>(randomNormal: [6]))))
    let id3 = x.store(hexagon[hexagonId[3]].retract(Vector6(s * Tensor<Double>(randomNormal: [6]))))
    let id4 = x.store(hexagon[hexagonId[4]].retract(Vector6(s * Tensor<Double>(randomNormal: [6]))))
    let id5 = x.store(hexagon[hexagonId[5]].retract(Vector6(s * Tensor<Double>(randomNormal: [6]))))
    
    var fg = FactorGraph()
    fg.store(PriorFactor3(id0, p0))
    let delta: Pose3 = between(p0, p1)

    fg.store(BetweenFactor3(id0, id1, delta))
    fg.store(BetweenFactor3(id1, id2, delta))
    fg.store(BetweenFactor3(id2, id3, delta))
    fg.store(BetweenFactor3(id3, id4, delta))
    fg.store(BetweenFactor3(id4, id5, delta))
    fg.store(BetweenFactor3(id5, id0, delta))

    // optimize
    for _ in 0..<16 {
      let gfg = fg.linearized(at: x)
      var dx = x.tangentVectorZeros
      var optimizer = GenericCGLS(precision: 1e-6, max_iteration: 500)
      optimizer.optimize(gfg: gfg, initial: &dx)
      x.move(along: (-1) * dx)
    }

    let pose_1 = x[id1]
    assertAllKeyPathEqual(pose_1, p1, accuracy: 1e-2)
  }
  
  func testGtsamPose3SLAMExampleChordal() {
    // Create a hexagon of poses
    let (hexagonId, hexagon) = circlePose3(numPoses: 6, radius: 1.0)
    let p0 = hexagon[hexagonId[0]]
    let p1 = hexagon[hexagonId[1]]
    let p2 = hexagon[hexagonId[2]]
    
    var x = VariableAssignments()
    
    let s = 0.90
    let id0 = x.store(p0)
    let id1 = x.store(hexagon[hexagonId[1]].retract(Vector6(s * Tensor<Double>(randomNormal: [6]))))
    let id2 = x.store(hexagon[hexagonId[2]].retract(Vector6(s * Tensor<Double>(randomNormal: [6]))))
    let id3 = x.store(hexagon[hexagonId[3]].retract(Vector6(s * Tensor<Double>(randomNormal: [6]))))
    let id4 = x.store(hexagon[hexagonId[4]].retract(Vector6(s * Tensor<Double>(randomNormal: [6]))))
    let id5 = x.store(hexagon[hexagonId[5]].retract(Vector6(s * Tensor<Double>(randomNormal: [6]))))
    
    var fg = FactorGraph()
    fg.store(PriorFactor3(id0, p0))
    let delta: Pose3 = between(p0, p1)

    fg.store(BetweenFactorAlternative3(id0, id1, delta))
    fg.store(BetweenFactorAlternative3(id1, id2, delta))
    fg.store(BetweenFactorAlternative3(id2, id3, delta))
    fg.store(BetweenFactorAlternative3(id3, id4, delta))
    fg.store(BetweenFactorAlternative3(id4, id5, delta))
    fg.store(BetweenFactorAlternative3(id5, id0, delta))

    // optimize
    for _ in 0..<16 {
      let gfg = fg.linearized(at: x)
      var dx = x.tangentVectorZeros
      var optimizer = GenericCGLS(precision: 1e-6, max_iteration: 500)
      optimizer.optimize(gfg: gfg, initial: &dx)
      x.move(along: (-1) * dx)
    }

    let pose_2 = x[id2]
    assertAllKeyPathEqual(pose_2, p2, accuracy: 1e-2)
  }
}
