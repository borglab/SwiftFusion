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
import SwiftFusion
import XCTest

final class GradientDescentTests: XCTestCase {
  /// Test convergence for a simple Pose2SLAM graph.
  // func testPose2SLAM() {
  //   var x = VariableAssignments()
  //   let pose1ID = x.store(Pose2(Rot2(0.2), Vector2(0.5, 0.0)))
  //   let pose2ID = x.store(Pose2(Rot2(-0.2), Vector2(2.3, 0.1)))
  //   let pose3ID = x.store(Pose2(Rot2(.pi / 2), Vector2(4.1, 0.1)))
  //   let pose4ID = x.store(Pose2(Rot2(.pi), Vector2(4.0, 2.0)))
  //   let pose5ID = x.store(Pose2(Rot2(-.pi / 2), Vector2(2.1, 2.1)))

  //   var graph = FactorGraph()
  //   graph.store(BetweenFactor(pose2ID, pose1ID, Pose2(2.0, 0.0, .pi / 2)))
  //   graph.store(BetweenFactor(pose3ID, pose2ID, Pose2(2.0, 0.0, .pi / 2)))
  //   graph.store(BetweenFactor(pose4ID, pose3ID, Pose2(2.0, 0.0, .pi / 2)))
  //   graph.store(BetweenFactor(pose5ID, pose4ID, Pose2(2.0, 0.0, .pi / 2)))
  //   graph.store(PriorFactor(pose1ID, Pose2(0, 0, 0)))

  //   let optimizer = GradientDescent(learningRate: 1e-2)
  //   for _ in 0..<10000 {
  //     optimizer.update(&x, objective: graph)
  //   }

  //   // Test condition: pose 5 should be identical to pose 1 (close loop).
  //   XCTAssertEqual(between(x[pose1ID], x[pose5ID]).t.norm, 0.0, accuracy: 1e-2)
  // }
}
