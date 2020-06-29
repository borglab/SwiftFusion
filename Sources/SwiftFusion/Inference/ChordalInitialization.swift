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

/// A BetweenFactor alternative that uses the Chordal (Frobenious) norm on rotation for Rot3
/// Please refer to Carlone15icra (Initialization Techniques for 3D SLAM: a Survey on Rotation Estimation and its Use in Pose Graph Optimization)
/// for explanation.
public struct FrobeniusFactorRot3: LinearizableFactor
{
  public typealias Variables = Tuple2<Vector9, Vector9>
  public typealias JacobianRows = Array9<Tuple2<Vector9.TangentVector, Vector9.TangentVector>>
  
  public let edges: Variables.Indices
  public let difference: Vector9

  public init(_ startId: TypedID<Vector9, Int>, _ endId: TypedID<Vector9, Int>, _ difference: Vector9) {
    self.edges = Tuple2(startId, endId)
    self.difference = difference
  }

  public typealias ErrorVector = Vector9
  
  @differentiable
  public func errorVector(_ start: Vector9, _ end: Vector9) -> ErrorVector {
    let R2 = Matrix3(end.s0, end.s1, end.s2, end.s3, end.s4, end.s5, end.s6, end.s7, end.s8)
    let R12 = Matrix3(difference.s0, difference.s1, difference.s2, difference.s3, difference.s4, difference.s5, difference.s6, difference.s7, difference.s8)
    let R = matmul(R12, R2.transposed()).transposed()
    let R_v = Vector9(R.s00, R.s01, R.s02, R.s10, R.s11, R.s12, R.s20, R.s21, R.s22)
    return R_v - start
  }

  // Note: All the remaining code in this factor is boilerplate that we can eventually eliminate
  // with sugar.
  
  public func error(at x: Variables) -> Double {
    return errorVector(at: x).squaredNorm
  }

  public func errorVector(at x: Variables) -> ErrorVector {
    return errorVector(x.head, x.tail.head)
  }

  public typealias Linearization = JacobianFactor<JacobianRows, ErrorVector>
  public func linearized(at x: Variables) -> Linearization {
    Linearization(linearizing: errorVector, at: x, edges: edges)
  }
}

/// A factor for the anchor in chordal initialization
public struct FrobeniusAnchorFactorRot3: LinearizableFactor
{
  public typealias Variables = Tuple1<Vector9>
  public typealias JacobianRows = Array9<Tuple1<Vector9.TangentVector>>
  
  public let edges: Variables.Indices
  public let prior: Vector9

  public init(_ id: TypedID<Vector9, Int>, _ val: Vector9) {
    self.edges = Tuple1(id)
    self.prior = val
  }

  public typealias ErrorVector = Vector9
  public func errorVector(_ val: Vector9) -> ErrorVector {
    val + prior
  }

  // Note: All the remaining code in this factor is boilerplate that we can eventually eliminate
  // with sugar.
  
  public func error(at x: Variables) -> Double {
    return errorVector(at: x).squaredNorm
  }

  public func errorVector(at x: Variables) -> ErrorVector {
    return errorVector(x.head)
  }

  public typealias Linearization = JacobianFactor<JacobianRows, ErrorVector>
  public func linearized(at x: Variables) -> Linearization {
    Linearization(linearizing: errorVector, at: x, edges: edges)
  }
}

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
  
  /// Construct the orientation graph with FrobeniusFactor s.
  public func solveOrientationGraph(
    g: FactorGraph,
    v: VariableAssignments,
    ids: Array<TypedID<Pose3, Int>>
  ) -> VariableAssignments {
    var orientationGraph = FactorGraph()
    var orientations = VariableAssignments()
    var associations = Dictionary<Int, TypedID<Vector9, Int>>()
    for i in ids {
      // let R = v[i].rot.coordinate.R
      associations[i.perTypeID] = orientations.store(Vector9(0, 0, 0, 0, 0, 0, 0, 0, 0))
    }
    
    associations[anchorId.perTypeID] = orientations.store(Vector9(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    
    for factor in g.factors(type: BetweenFactor3.self) {
      let R = factor.difference.rot.coordinate.R
      let R_v = Vector9(R.s00, R.s01, R.s02, R.s10, R.s11, R.s12, R.s20, R.s21, R.s22)
      let frob_factor = FrobeniusFactorRot3(associations[factor.edges.head.perTypeID]!, associations[factor.edges.tail.head.perTypeID]!, R_v)
      orientationGraph.store(frob_factor)
    }
    
    orientationGraph.store(FrobeniusAnchorFactorRot3(associations[anchorId.perTypeID]!, Vector9(1.0, 0.0, 0.0, /*  */ 0.0, 1.0, 0.0, /*  */ 0.0, 0.0, 1.0)))
    
    var optimizer = GenericCGLS()
    let linearGraph = orientationGraph.linearized(at: orientations)
    optimizer.optimize(gfg: linearGraph, initial: &orientations)
    
    return normalizeRelaxedRotations(orientations, associations: associations, ids: ids)
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
      
      let M = Matrix3(M_v.s0, M_v.s1, M_v.s2, M_v.s3, M_v.s4, M_v.s5, M_v.s6, M_v.s7, M_v.s8)
      
      let initRot = Rot3.ClosestTo(mat: M)
      
      // TODO(fan): relies on the assumption of continous and ordered allocation
      let _ = validRot3.store(initRot)
    }
    return validRot3;
  }
  
  /// This function computes the inital poses given the chordal initialized rotations.
  public func computePoses(graph: FactorGraph, orientations: VariableAssignments, ids: Array<TypedID<Pose3, Int>>) -> VariableAssignments {
    var val = VariableAssignments()
    for v in ids {
      let _ = val.store(Pose3(orientations[TypedID<Rot3, Int>(v.perTypeID)], Vector3(0,0,0)))
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
  /// TODO(fan): This function builds upon the assumption that all variables stored are Pose3s, could fail if that is not the case.
  public mutating func GetInitializations(graph: FactorGraph, val: VariableAssignments, ids: Array<TypedID<Pose3, Int>>) -> VariableAssignments {
    var val_copy = val
    anchorId = val_copy.store(Pose3())
    // We "extract" the Pose3 subgraph of the original graph: this
    // is done to properly model priors and avoiding operating on a larger graph
    let pose3Graph = buildPose3graph(graph: graph)

    // Get orientations from relative orientation measurements
    let orientations = solveOrientationGraph(g: pose3Graph, v: val_copy, ids: ids)
    
    // Compute the full poses (1 GN iteration on full poses)
    return computePoses(graph: pose3Graph, orientations: orientations, ids: ids)
  }
}
