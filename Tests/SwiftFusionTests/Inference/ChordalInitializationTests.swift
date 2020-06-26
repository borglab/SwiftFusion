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

let x0 = TypedID<Pose3, Int>(0)
let x1 = TypedID<Pose3, Int>(1)
let x2 = TypedID<Pose3, Int>(2)
let x3 = TypedID<Pose3, Int>(3)

let  p0 = Vector3(0,0,0);
let  R0 = Rot3.fromTangent(Vector3(0.0,0.0,0.0))
let  p1 = Vector3(1,2,0)
let  R1 = Rot3.fromTangent(Vector3(0.0,0.0,1.570796))
let  p2 = Vector3(0,2,0)
let  R2 = Rot3.fromTangent(Vector3(0.0,0.0,3.141593))
let  p3 = Vector3(-1,1,0)
let  R3 = Rot3.fromTangent(Vector3(0.0,0.0,4.712389))

let pose0 = Pose3(R0,p0)
let pose1 = Pose3(R1,p1)
let pose2 = Pose3(R2,p2)
let pose3 = Pose3(R3,p3)

func graph1() -> FactorGraph {
  var g = FactorGraph()
  g.store(BetweenFactor3(x0, x1, between(pose0, pose1)))
  g.store(BetweenFactor3(x1, x2, between(pose1, pose2)))
  g.store(BetweenFactor3(x2, x3, between(pose2, pose3)))
  g.store(BetweenFactor3(x2, x0, between(pose2, pose0)))
  g.store(BetweenFactor3(x0, x3, between(pose0, pose3)))
  g.store(PriorFactor3(x0, pose0))
  return g
}

class ChordalInitializationTests: XCTestCase {
  func testChordalOrientation() {
    var ci = ChordalInitialization()
    
    var val = VariableAssignments()
    val.store(pose0)
    val.store(pose1)
    val.store(pose2)
    val.store(pose3)
    
    var val_copy = val
    ci.anchorId = val_copy.store(Pose3())
    
    let pose3Graph = ci.buildPose3graph(graph: graph1())

    let initial = ci.computeOrientationsChordal(graph: pose3Graph, val: val_copy, ids: [x0, x1, x2, x3])

    // comparison is up to M_PI, that's why we add some multiples of 2*M_PI
    assertAllKeyPathEqual( R0, initial[TypedID<Rot3, Int>(x0.perTypeID)], accuracy: 1e-6)
    assertAllKeyPathEqual( R1, initial[TypedID<Rot3, Int>(x1.perTypeID)], accuracy: 1e-6)
    assertAllKeyPathEqual( R2, initial[TypedID<Rot3, Int>(x2.perTypeID)], accuracy: 1e-6)
    assertAllKeyPathEqual( R3, initial[TypedID<Rot3, Int>(x3.perTypeID)], accuracy: 1e-6)
  }
}
