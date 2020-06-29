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

import PenguinStructures
import TensorFlow

/// Type shorthands used in the relaxed pose graph
public typealias Jacobian9x9_1 = Array9<Tuple1<Vector9>>
public typealias Jacobian9x9_2 = Array9<Tuple2<Vector9, Vector9>>
public typealias JacobianFactor9x9_1 = JacobianFactor<Jacobian9x9_1, Vector9>
public typealias JacobianFactor9x9_2 = JacobianFactor<Jacobian9x9_2, Vector9>

/// Chordal Initialization for Pose3s
public struct ChordalInitialization {
  public var anchorId: TypedID<Pose3, Int>
  
  public init() {
    anchorId = TypedID<Pose3, Int>(0)
  }
  
  /// Extract a subgraph of the original graph with only Pose3s.
  public func buildPose3graph(graph: FactorGraph) -> FactorGraph {
    var pose3Graph = FactorGraph()

    for factor in graph.factors(type: BetweenFactor3.self) {
        pose3Graph.store(factor)
    }
    
    for factor in graph.factors(type: PriorFactor3.self) {
      pose3Graph.store(BetweenFactor3(anchorId, factor.edges.head, factor.prior))
    }
    
    return pose3Graph
  }
  
  /// Construct the orientation graph (unconstrained version of the between in Frobenius norm).
  public func buildLinearOrientationGraph(
    g: FactorGraph,
    v: VariableAssignments,
    v_linear: VariableAssignments,
    associations: Dictionary<Int, TypedID<Vector9, Int>>
  ) -> GaussianFactorGraph {
    var linearGraph = GaussianFactorGraph(zeroValues: v_linear.tangentVectorZeros)

    for pose3Between in g.factors(type: BetweenFactor3.self) {
      var Rij = Matrix3.Identity

      Rij = pose3Between.difference.rot.coordinate.R
      
      let key1 = associations[pose3Between.edges.head.perTypeID]!
      let key2 = associations[pose3Between.edges.tail.head.perTypeID]!
      
      let M9: Jacobian9x9_2 = Array9([
        Tuple2(Vector9(-1,0,0,0,0,0,0,0,0), Vector9(Rij.s00,Rij.s01,Rij.s02,0,0,0,0,0,0)),
        Tuple2(Vector9(0,-1,0,0,0,0,0,0,0), Vector9(Rij.s10,Rij.s11,Rij.s12,0,0,0,0,0,0)),
        Tuple2(Vector9(0,0,-1,0,0,0,0,0,0), Vector9(Rij.s20,Rij.s21,Rij.s22,0,0,0,0,0,0)),
        Tuple2(Vector9(0,0,0,-1,0,0,0,0,0), Vector9(0,0,0,Rij.s00,Rij.s01,Rij.s02,0,0,0)),
        Tuple2(Vector9(0,0,0,0,-1,0,0,0,0), Vector9(0,0,0,Rij.s10,Rij.s11,Rij.s12,0,0,0)),
        Tuple2(Vector9(0,0,0,0,0,-1,0,0,0), Vector9(0,0,0,Rij.s20,Rij.s21,Rij.s22,0,0,0)),
        Tuple2(Vector9(0,0,0,0,0,0,-1,0,0), Vector9(0,0,0,0,0,0,Rij.s00,Rij.s01,Rij.s02)),
        Tuple2(Vector9(0,0,0,0,0,0,0,-1,0), Vector9(0,0,0,0,0,0,Rij.s10,Rij.s11,Rij.s12)),
        Tuple2(Vector9(0,0,0,0,0,0,0,0,-1), Vector9(0,0,0,0,0,0,Rij.s20,Rij.s21,Rij.s22))
      ])
      
      let b = Vector9(0, 0, 0, 0, 0, 0, 0, 0, 0)
      
      linearGraph.store(JacobianFactor9x9_2(jacobian: M9, error: b, edges: Tuple2(key1, key2)))
    }
    
    let I_9x9: Jacobian9x9_1 = Array9([
      Tuple1(Vector9(1,0,0,0,0,0,0,0,0)),
      Tuple1(Vector9(0,1,0,0,0,0,0,0,0)),
      Tuple1(Vector9(0,0,1,0,0,0,0,0,0)),
      Tuple1(Vector9(0,0,0,1,0,0,0,0,0)),
      Tuple1(Vector9(0,0,0,0,1,0,0,0,0)),
      Tuple1(Vector9(0,0,0,0,0,1,0,0,0)),
      Tuple1(Vector9(0,0,0,0,0,0,1,0,0)),
      Tuple1(Vector9(0,0,0,0,0,0,0,1,0)),
      Tuple1(Vector9(0,0,0,0,0,0,0,0,1))
    ])
    
    // prior on the anchor orientation
    linearGraph.store(JacobianFactor9x9_1(jacobian: I_9x9,
                                          error: Vector9(1.0, 0.0, 0.0, /*  */ 0.0, 1.0, 0.0, /*  */ 0.0, 0.0, 1.0),
                                          edges: Tuple1(associations[anchorId.perTypeID]!)))
    
    return linearGraph
  }
  
  /// This function finds the closest Rot3 to the unconstrained 3x3 matrix with SVD.
  /// TODO(fan): replace this with a 3x3 specialized SVD instead of this generic SVD (slow)
  public func normalizeRelaxedRotations(
    _ relaxedRot3: VariableAssignments,
    associations: Dictionary<Int, TypedID<Vector9, Int>>,
    ids: Array<TypedID<Pose3, Int>>) -> VariableAssignments {
    var validRot3 = VariableAssignments()
    
    for v in ids {
      let M_v: Vector9 = relaxedRot3[associations[v.perTypeID]!]
      
      let M = M_v.tensor.reshaped(to: [3, 3])
      
      let (_, U, V) = M.transposed().svd(computeUV: true, fullMatrices: true)
      let UVT: Tensor<Double> = matmul(U!, V!.transposed())
      
      let det = (UVT[0, 0].scalar! * (UVT[1, 1].scalar! * UVT[2, 2].scalar! - UVT[2, 1].scalar! * UVT[1, 2].scalar!)
                 - UVT[1, 0].scalar! * (UVT[0, 1].scalar! * UVT[2, 2].scalar! - UVT[2, 1].scalar! * UVT[0, 2].scalar!)
                 + UVT[2, 0].scalar! * (UVT[0, 1].scalar! * UVT[1, 2].scalar! - UVT[1, 1].scalar! * UVT[0, 2].scalar!))
      
      let R = matmul(matmul(U!, Tensor<Double>(shape: [3], scalars: [1, 1, det]).diagonal()), V!.transposed()).transposed()
      
      let initRot = Rot3(coordinate:
                          Matrix3Coordinate(Matrix3(R[0, 0].scalar!, R[0, 1].scalar!, R[0, 2].scalar!,
                                                    R[1, 0].scalar!, R[1, 1].scalar!, R[1, 2].scalar!,
                                                    R[2, 0].scalar!, R[2, 1].scalar!, R[2, 2].scalar!))
      )
      
      // TODO(fan): relies on the assumption of continous and ordered allocation
      let _ = validRot3.store(initRot)
    }
    return validRot3;
  }
  
  /// This function computes the rotations
  public func computeOrientationsChordal(graph: FactorGraph, val: VariableAssignments, ids: Array<TypedID<Pose3, Int>>) -> VariableAssignments {
    var relaxedRot3 = VariableAssignments()
    var associations = Dictionary<Int, TypedID<Vector9, Int>>()
    
    for v in ids {
      let R = val[v].rot.coordinate.R
      associations[v.perTypeID] = relaxedRot3.store(
        Vector9(R.s00, R.s01, R.s02,
        R.s10, R.s11, R.s12,
        R.s20, R.s21, R.s22))
    }
    
    let R_a = val[anchorId].rot.coordinate.R
    associations[anchorId.perTypeID] = relaxedRot3.store(Vector9(R_a.s00, R_a.s01, R_a.s02,
    R_a.s10, R_a.s11, R_a.s12,
    R_a.s20, R_a.s21, R_a.s22))
    
    // regularize measurements and plug everything in a factor graph
    let relaxedGraph: GaussianFactorGraph = buildLinearOrientationGraph(g: graph, v: val, v_linear: relaxedRot3, associations: associations)

    // Solve the LFG
    var optimizer = GenericCGLS()
    print("[COC BEFORE] \(relaxedGraph.errorVectors(at: relaxedRot3).squaredNorm)")
    optimizer.optimize(gfg: relaxedGraph, initial: &relaxedRot3)
    print("[COC AFTER ] \(relaxedGraph.errorVectors(at: relaxedRot3).squaredNorm)")
    
    for i in associations.sorted(by: { $0.0 < $1.0 }) {
      print("\(i.key): \(relaxedRot3[i.value])")
    }
    
    print("\(ids)")
    let result = normalizeRelaxedRotations(relaxedRot3, associations: associations, ids: ids)
    
    // normalize and compute Rot3
    return result
  }
  
  /// This function computes the inital poses given the chordal initialized rotations.
  public func computePoses(graph: FactorGraph, original_initialization: VariableAssignments, orientations: VariableAssignments, ids: Array<TypedID<Pose3, Int>>) -> VariableAssignments {
    var val = original_initialization
    for v in ids {
      val[v]=Pose3(orientations[TypedID<Rot3, Int>(v.perTypeID)], Vector3(0,0,0))
    }
    
    // optimize for 1 G-N iteration
    for _ in 0..<1 {
      let gfg = graph.linearized(at: val)
      var dx = val.tangentVectorZeros
      var optimizer = GenericCGLS(precision: 1e-6, max_iteration: 500)
      optimizer.optimize(gfg: gfg, initial: &dx)
      val.move(along: (-1) * dx)
    }
    return val
  }
  
  /// This function computes the chordal initialization. Normally this is what the user needs to call.
  public mutating func GetInitializations(graph: FactorGraph, val: VariableAssignments, ids: Array<TypedID<Pose3, Int>>) -> VariableAssignments {
    var val_copy = val
    anchorId = val_copy.store(Pose3())
    // We "extract" the Pose3 subgraph of the original graph: this
    // is done to properly model priors and avoiding operating on a larger graph
    let pose3Graph = buildPose3graph(graph: graph)

    // Get orientations from relative orientation measurements
    let orientations = computeOrientationsChordal(graph: pose3Graph, val: val_copy, ids: ids)

    // Compute the full poses (1 GN iteration on full poses)
    return computePoses(graph: pose3Graph, original_initialization: val_copy, orientations: orientations, ids: ids)
  }
}
