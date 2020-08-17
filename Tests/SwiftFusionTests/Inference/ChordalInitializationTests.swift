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

// MARK: sample data

// symbol shorthands
let x0 = TypedID<Pose3>(0)
let x1 = TypedID<Pose3>(1)
let x2 = TypedID<Pose3>(2)
let x3 = TypedID<Pose3>(3)

// ground truth
let  p0 = Vector3(0,0,0);
let  R0 = Rot3.fromTangent(Vector3(0.0,0.0,0.0))
let  p1 = Vector3(1,2,0)
// let  R1 = Rot3.fromTangent(Vector3(0.0,0.0,1.570796))
let R1 = Rot3(3.26795e-07, -1, 0,
              1, 3.26795e-07,  0,
              0,           0,  1)
let  p2 = Vector3(0,2,0)
let  R2 = Rot3.fromTangent(Vector3(0.0,0.0,3.141593))
let  p3 = Vector3(-1,1,0)
let  R3 = Rot3.fromTangent(Vector3(0.0,0.0,4.712389))

let pose0 = Pose3(R0,p0)
let pose1 = Pose3(R1,p1)
let pose2 = Pose3(R2,p2)
let pose3 = Pose3(R3,p3)

/// simple test graph for the chordal initialization
func graph1() -> FactorGraph {
  var g = FactorGraph()
  g.store(BetweenFactor(x0, x1, between(pose0, pose1)))
  g.store(BetweenFactor(x1, x2, between(pose1, pose2)))
  g.store(BetweenFactor(x2, x3, between(pose2, pose3)))
  g.store(BetweenFactor(x2, x0, between(pose2, pose0)))
  g.store(BetweenFactor(x0, x3, between(pose0, pose3)))
  g.store(PriorFactor(x0, pose0))
  return g
}

class ChordalInitializationTests: XCTestCase {
  /// make sure the derivatives are correct
  func testFrobeniusRot3BetweenJacobians() {
    
    print(Matrix3.standardBasis)
    
    var val = VariableAssignments()
    let p0 = val.store(Matrix3(1,0,0,0,1,0,0,0,1))
    let p1 = val.store(Matrix3(1,0,0,0,1,0,0,0,1))
    
    let frf = RelaxedRotationFactorRot3(p0, p1, Matrix3(0.0,0.1,0.2,1.0,1.1,1.2,2.0,2.1,2.2))
    
    let frf_j = JacobianFactor9x3x3_2(linearizing: frf, at: Tuple2(val[p0], val[p1]))
    
    let Rij = Matrix3(0.0,0.1,0.2,1.0,1.1,1.2,2.0,2.1,2.2)
    let M9: Jacobian9x3x3_2 = [
      Tuple2(Matrix3(-1,0,0,0,0,0,0,0,0), Matrix3(Rij[0, 0],Rij[0, 1],Rij[0, 2],0,0,0,0,0,0)),
      Tuple2(Matrix3(0,-1,0,0,0,0,0,0,0), Matrix3(Rij[1, 0],Rij[1, 1],Rij[1, 2],0,0,0,0,0,0)),
      Tuple2(Matrix3(0,0,-1,0,0,0,0,0,0), Matrix3(Rij[2, 0],Rij[2, 1],Rij[2, 2],0,0,0,0,0,0)),
      Tuple2(Matrix3(0,0,0,-1,0,0,0,0,0), Matrix3(0,0,0,Rij[0, 0],Rij[0, 1],Rij[0, 2],0,0,0)),
      Tuple2(Matrix3(0,0,0,0,-1,0,0,0,0), Matrix3(0,0,0,Rij[1, 0],Rij[1, 1],Rij[1, 2],0,0,0)),
      Tuple2(Matrix3(0,0,0,0,0,-1,0,0,0), Matrix3(0,0,0,Rij[2, 0],Rij[2, 1],Rij[2, 2],0,0,0)),
      Tuple2(Matrix3(0,0,0,0,0,0,-1,0,0), Matrix3(0,0,0,0,0,0,Rij[0, 0],Rij[0, 1],Rij[0, 2])),
      Tuple2(Matrix3(0,0,0,0,0,0,0,-1,0), Matrix3(0,0,0,0,0,0,Rij[1, 0],Rij[1, 1],Rij[1, 2])),
      Tuple2(Matrix3(0,0,0,0,0,0,0,0,-1), Matrix3(0,0,0,0,0,0,Rij[2, 0],Rij[2, 1],Rij[2, 2]))
    ]
    
    let b = Vector9(0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    let jf = JacobianFactor9x3x3_2(jacobian: M9, error: b, edges: Tuple2(TypedID<Matrix3>(0), TypedID<Matrix3>(1)))
    
    // assert the jacobian is correct
    assertEqual(
      Tensor<Double>(stacking: frf_j.jacobian.flatMap { [$0.head.tensor, $0.tail.head.tensor] }),
      Tensor<Double>(stacking: jf.jacobian.flatMap { [$0.head.tensor, $0.tail.head.tensor] })
      , accuracy: 1e-4
    )
    
    let frf_zero = RelaxedRotationFactorRot3(p0, p1, Matrix3.identity)
    let frf_zero_j = JacobianFactor9x3x3_2(linearizing: frf_zero, at: Tuple2(val[p0], val[p1]))
    
    // assert the zero error is correct
    assertAllKeyPathEqual(frf_zero_j.error, jf.error, accuracy: 1e-5)
    
    let fpf = RelaxedAnchorFactorRot3(p0, Matrix3.identity)
    
    let fpf_j = JacobianFactor9x3x3_1(linearizing: fpf, at: Tuple1(Matrix3.zero))
    
    let I_9x9: Jacobian9x3x3_1 = [
      Tuple1(Matrix3(1,0,0,0,0,0,0,0,0)),
      Tuple1(Matrix3(0,1,0,0,0,0,0,0,0)),
      Tuple1(Matrix3(0,0,1,0,0,0,0,0,0)),
      Tuple1(Matrix3(0,0,0,1,0,0,0,0,0)),
      Tuple1(Matrix3(0,0,0,0,1,0,0,0,0)),
      Tuple1(Matrix3(0,0,0,0,0,1,0,0,0)),
      Tuple1(Matrix3(0,0,0,0,0,0,1,0,0)),
      Tuple1(Matrix3(0,0,0,0,0,0,0,1,0)),
      Tuple1(Matrix3(0,0,0,0,0,0,0,0,1))
    ]
    
    // prior on the anchor orientation
    let jf_p = JacobianFactor9x3x3_1(jacobian: I_9x9,
                                          error: Vector9(1.0, 0.0, 0.0, /*  */ 0.0, 1.0, 0.0, /*  */ 0.0, 0.0, 1.0),
                                          edges: Tuple1(TypedID<Matrix3>(0)))
    
    // assert the Jacobian is correct
    assertEqual(
      Tensor<Double>(stacking: frf_j.jacobian.flatMap { [$0.head.tensor, $0.tail.head.tensor] }),
      Tensor<Double>(stacking: jf.jacobian.flatMap { [$0.head.tensor, $0.tail.head.tensor] })
      , accuracy: 1e-4
    )
    
    // assert the error at zero is correct
    assertAllKeyPathEqual(fpf_j.error, jf_p.error, accuracy: 1e-5)
  }
  
  /// sanity test for the chordal initialization on `graph1`
  func testChordalOrientation() {
    var ci = ChordalInitialization()
    
    var val = VariableAssignments()
    let _ = val.store(pose0)
    let _ = val.store(pose0)
    let _ = val.store(pose0)
    let _ = val.store(pose0)
    
    var val_copy = val
    ci.anchorId = val_copy.store(Pose3())
    
    let pose3Graph = ci.buildPose3graph(graph: graph1())
    
    let initial = ci.solveOrientationGraph(g: pose3Graph, v: val_copy, ids: [x0, x1, x2, x3])
    
    assertAllKeyPathEqual(Matrix3.identity, initial[TypedID<Rot3>(ci.anchorId.perTypeID)].coordinate.R, accuracy: 1e-5)
    
    assertAllKeyPathEqual( R0, initial[TypedID<Rot3>(x0.perTypeID)], accuracy: 1e-5)
    assertAllKeyPathEqual( R1, initial[TypedID<Rot3>(x1.perTypeID)], accuracy: 1e-5)
    assertAllKeyPathEqual( R2, initial[TypedID<Rot3>(x2.perTypeID)], accuracy: 1e-5)
    assertAllKeyPathEqual( R3, initial[TypedID<Rot3>(x3.perTypeID)], accuracy: 1e-5)
  }
}
