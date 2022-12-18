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
import SwiftFusion

class FactorGraphTests: XCTestCase {
  /// Tests that we can read factors from a factor graph.
  func testGetFactors() {
    let pose1ID = TypedID<Pose2>(0)
    let pose2ID = TypedID<Pose2>(1)

    var graph = FactorGraph()
    graph.store(PriorFactor(pose1ID, Pose2(1, 2, 3)))
    graph.store(PriorFactor(pose2ID, Pose2(4, 5, 6)))
    graph.store(BetweenFactor(pose1ID, pose2ID, Pose2(7, 8, 9)))

    XCTAssertEqual(
      graph.factors(type: PriorFactor<Pose2>.self).map { $0.edges },
      [Tuple1(pose1ID), Tuple1(pose2ID)]
    )
    XCTAssertEqual(
      graph.factors(type: PriorFactor<Pose2>.self).map { $0.prior },
      [Pose2(1, 2, 3), Pose2(4, 5, 6)]
    )

    XCTAssertEqual(
      graph.factors(type: BetweenFactor<Pose2>.self).map { $0.edges },
      [Tuple2(pose1ID, pose2ID)]
    )
    XCTAssertEqual(
      graph.factors(type: BetweenFactor<Pose2>.self).map { $0.difference },
      [Pose2(7, 8, 9)]
    )

    XCTAssertEqual(graph.factors(type: PriorFactor<Pose3>.self).count, 0)
  }

  func testSimplePose2SLAM() {
    var x = VariableAssignments()
    let pose1ID = x.store(Pose2(Rot2(0.2), Vector2(0.5, 0.0)))
    let pose2ID = x.store(Pose2(Rot2(-0.2), Vector2(2.3, 0.1)))
    let pose3ID = x.store(Pose2(Rot2(.pi / 2), Vector2(4.1, 0.1)))
    let pose4ID = x.store(Pose2(Rot2(.pi), Vector2(4.0, 2.0)))
    let pose5ID = x.store(Pose2(Rot2(-.pi / 2), Vector2(2.1, 2.1)))

    var graph = FactorGraph()
    graph.store(BetweenFactor(pose2ID, pose1ID, Pose2(2.0, 0.0, .pi / 2)))
    graph.store(BetweenFactor(pose3ID, pose2ID, Pose2(2.0, 0.0, .pi / 2)))
    graph.store(BetweenFactor(pose4ID, pose3ID, Pose2(2.0, 0.0, .pi / 2)))
    graph.store(BetweenFactor(pose5ID, pose4ID, Pose2(2.0, 0.0, .pi / 2)))
    graph.store(PriorFactor(pose1ID, Pose2(0, 0, 0)))

    for _ in 0..<5 {
      let linearized = graph.linearized(at: x)
      var dx = x.tangentVectorZeros
      var optimizer = GenericCGLS(precision: 1e-6, max_iteration: 500)
      optimizer.optimize(gfg: linearized, initial: &dx)
      x.move(along: dx)
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
  func circlePose3(numPoses: Int = 8, radius: Double = 1.0) -> (Array<TypedID<Pose3>>, VariableAssignments) {
    var ids: Array<TypedID<Pose3>> = []
    var values = VariableAssignments()
    var theta: Double = 0.0
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
    
    let s: Double = 0.10
    let id0 = x.store(p0)
    let id1 = x.store(hexagon[hexagonId[1]].retract(Vector6(flatTensor: s * Tensor<Double>(randomNormal: [6]))))
    let id2 = x.store(hexagon[hexagonId[2]].retract(Vector6(flatTensor: s * Tensor<Double>(randomNormal: [6]))))
    let id3 = x.store(hexagon[hexagonId[3]].retract(Vector6(flatTensor: s * Tensor<Double>(randomNormal: [6]))))
    let id4 = x.store(hexagon[hexagonId[4]].retract(Vector6(flatTensor: s * Tensor<Double>(randomNormal: [6]))))
    let id5 = x.store(hexagon[hexagonId[5]].retract(Vector6(flatTensor: s * Tensor<Double>(randomNormal: [6]))))
    
    var fg = FactorGraph()
    fg.store(PriorFactor(id0, p0))
    let delta: Pose3 = between(p0, p1)

    fg.store(BetweenFactor(id0, id1, delta))
    fg.store(BetweenFactor(id1, id2, delta))
    fg.store(BetweenFactor(id2, id3, delta))
    fg.store(BetweenFactor(id3, id4, delta))
    fg.store(BetweenFactor(id4, id5, delta))
    fg.store(BetweenFactor(id5, id0, delta))

    // optimize
    for _ in 0..<16 {
      let gfg = fg.linearized(at: x)
      var dx = x.tangentVectorZeros
      var optimizer = GenericCGLS(precision: 1e-6, max_iteration: 500)
      optimizer.optimize(gfg: gfg, initial: &dx)
      x.move(along: dx)
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

    // Sometimes this optimization gets stuck in a local minimum, so attempt until we find the
    // global minimum, with a maximum of `attemptCount` attempts.
    let attemptCount = 10
    for _ in 0..<attemptCount {
      var x = VariableAssignments()
      
      let s: Double = 0.9
      let id0 = x.store(p0)
      let id1 = x.store(hexagon[hexagonId[1]].retract(Vector6(flatTensor: s * Tensor(randomNormal: [6]))))
      let id2 = x.store(hexagon[hexagonId[2]].retract(Vector6(flatTensor: s * Tensor(randomNormal: [6]))))
      let id3 = x.store(hexagon[hexagonId[3]].retract(Vector6(flatTensor: s * Tensor(randomNormal: [6]))))
      let id4 = x.store(hexagon[hexagonId[4]].retract(Vector6(flatTensor: s * Tensor(randomNormal: [6]))))
      let id5 = x.store(hexagon[hexagonId[5]].retract(Vector6(flatTensor: s * Tensor(randomNormal: [6]))))
      
      var fg = FactorGraph()
      fg.store(PriorFactor(id0, p0))
      let delta: Pose3 = between(p0, p1)

      fg.store(BetweenFactorAlternative(id0, id1, delta))
      fg.store(BetweenFactorAlternative(id1, id2, delta))
      fg.store(BetweenFactorAlternative(id2, id3, delta))
      fg.store(BetweenFactorAlternative(id3, id4, delta))
      fg.store(BetweenFactorAlternative(id4, id5, delta))
      fg.store(BetweenFactorAlternative(id5, id0, delta))

      // optimize
      for _ in 0..<16 {
        let gfg = fg.linearized(at: x)
        var dx = x.tangentVectorZeros
        var optimizer = GenericCGLS(precision: 1e-6, max_iteration: 500)
        optimizer.optimize(gfg: gfg, initial: &dx)
        x.move(along: dx)
      }

      if fg.error(at: x) < 1e-5 {
        // Successfully found the global minimum.
        let pose_2 = x[id2]
        assertAllKeyPathEqual(pose_2, p2, accuracy: 1e-2)
        return
      }
    }

    XCTFail("failed to find the global minimum after \(attemptCount) attempts")
  }
  
  func testGtsamPose3SLAMExampleChordalInit() {
    // Create a hexagon of poses
    let (hexagonId, hexagon) = circlePose3(numPoses: 6, radius: 1.0)
    let p0 = hexagon[hexagonId[0]]
    let p1 = hexagon[hexagonId[1]]
    let p2 = hexagon[hexagonId[2]]

    // Sometimes this optimization gets stuck in a local minimum, so attempt until we find the
    // global minimum, with a maximum of `attemptCount` attempts.
    let attemptCount = 10
    for _ in 0..<attemptCount {
      var x = VariableAssignments()
      
      let s = 0.9
      let id0 = x.store(p0)
      let id1 = x.store(hexagon[hexagonId[1]].retract(Vector6(flatTensor: s * Tensor(randomNormal: [6]))))
      let id2 = x.store(hexagon[hexagonId[2]].retract(Vector6(flatTensor: s * Tensor(randomNormal: [6]))))
      let id3 = x.store(hexagon[hexagonId[3]].retract(Vector6(flatTensor: s * Tensor(randomNormal: [6]))))
      let id4 = x.store(hexagon[hexagonId[4]].retract(Vector6(flatTensor: s * Tensor(randomNormal: [6]))))
      let id5 = x.store(hexagon[hexagonId[5]].retract(Vector6(flatTensor: s * Tensor(randomNormal: [6]))))
      
      var fg = FactorGraph()
      fg.store(PriorFactor(id0, p0))
      let delta: Pose3 = between(p0, p1)

      fg.store(BetweenFactor(id0, id1, delta))
      fg.store(BetweenFactor(id1, id2, delta))
      fg.store(BetweenFactor(id2, id3, delta))
      fg.store(BetweenFactor(id3, id4, delta))
      fg.store(BetweenFactor(id4, id5, delta))
      fg.store(BetweenFactor(id5, id0, delta))

      let chordal_init = ChordalInitialization.GetInitializations(graph: fg, ids: [id0, id1, id2, id3, id4, id5])
      
      x = chordal_init
      
      // optimize
      for _ in 0..<16 {
        let gfg = fg.linearized(at: x)
        var dx = x.tangentVectorZeros
        var optimizer = GenericCGLS(precision: 1e-6, max_iteration: 500)
        optimizer.optimize(gfg: gfg, initial: &dx)
        x.move(along: dx)
      }

      if fg.error(at: x) < 1e-5 {
        // Successfully found the global minimum.
        let pose_2 = x[id2]
        assertAllKeyPathEqual(pose_2, p2, accuracy: 1e-2)
        return
      }
    }

    XCTFail("failed to find the global minimum after \(attemptCount) attempts")
  }

  /// Test the gradient of the error of a factor graph.
  func testGradient() {
    var vars = VariableAssignments()
    let v1ID = vars.store(Vector2(1, 2))
    let v2ID = vars.store(Vector2(3, 4))
    let v3ID = vars.store(Vector3(5, 6, 7))

    var graph = FactorGraph()
    graph.store(ScalarJacobianFactor(edges: Tuple1(v1ID), scalar: 1))
    graph.store(ScalarJacobianFactor(edges: Tuple1(v1ID), scalar: 2))
    graph.store(ScalarJacobianFactor(edges: Tuple1(v2ID), scalar: 5))
    graph.store(ScalarJacobianFactor(edges: Tuple1(v3ID), scalar: 10))

    let grad = graph.errorGradient(at: vars)

    // gradient of ||1 * v1||^2 + ||2 * v1||^2 at v1 = (1, 2)
    XCTAssertEqual(grad[v1ID], Vector2(10, 20))

    // gradient of ||5 * v2||^2 at v2 = (3, 4)
    XCTAssertEqual(grad[v2ID], Vector2(150, 200))

    // gradient of ||10 * v3||^2 at v3 = (5, 6, 7)
    XCTAssertEqual(grad[v3ID], Vector3(1000, 1200, 1400))
  }
}
