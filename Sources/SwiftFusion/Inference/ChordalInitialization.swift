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

/// A relaxed version of the Rot3 between that uses the Chordal (Frobenious) norm on rotation
/// Please refer to Carlone15icra (Initialization Techniques for 3D SLAM: a Survey on Rotation Estimation and its Use in Pose Graph Optimization)
/// for explanation.
public struct RelaxedRotationFactorRot3: LinearizableFactor2
{
  public let edges: Variables.Indices
  public let difference: Matrix3
  
  public init(_ id1: TypedID<Matrix3>, _ id2: TypedID<Matrix3>, _ difference: Matrix3) {
    self.edges = Tuple2(id1, id2)
    self.difference = difference
  }
  
  public typealias ErrorVector = Vector9
  
  @differentiable
  public func errorVector(_ R1: Matrix3, _ R2: Matrix3) -> ErrorVector {
    let R12 = difference
    let R1h = matmul(R12, R2.transposed()).transposed()
    return ErrorVector(R1h - R1)
  }
}

/// A factor for the anchor in chordal initialization
public struct RelaxedAnchorFactorRot3: LinearizableFactor1
{
  public let edges: Variables.Indices
  public let prior: Matrix3
  
  public init(_ id: TypedID<Matrix3>, _ prior: Matrix3) {
    self.edges = Tuple1(id)
    self.prior = prior
  }
  
  public typealias ErrorVector = Vector9

  @differentiable
  public func errorVector(_ val: Matrix3) -> ErrorVector {
    ErrorVector(val - prior)
  }
}

/// Type shorthands used in the relaxed pose graph
/// NOTE: Specializations are added in `FactorsStorage.swift`
public typealias Jacobian9x3x3_1 = Array9<Tuple1<Matrix3>>
public typealias Jacobian9x3x3_2 = Array9<Tuple2<Matrix3, Matrix3>>
public typealias JacobianFactor9x3x3_1 = JacobianFactor<Jacobian9x3x3_1, Vector9>
public typealias JacobianFactor9x3x3_2 = JacobianFactor<Jacobian9x3x3_2, Vector9>

/// Chordal Initialization for Pose3s
public struct ChordalInitialization {
  /// ID of the anchor used in chordal initialization, should only be used if not using `GetInitializations`.
  public var anchorId: TypedID<Pose3>
  
  public init() {
    anchorId = TypedID<Pose3>(0)
  }
  
  /// Extract a subgraph of the original graph with only Pose3s.
  public func buildPose3graph(graph: FactorGraph) -> FactorGraph {
    var pose3Graph = FactorGraph()
    
    for factor in graph.factors(type: BetweenFactor<Pose3>.self) {
      pose3Graph.store(factor)
    }
    
    for factor in graph.factors(type: PriorFactor<Pose3>.self) {
      pose3Graph.store(BetweenFactor(anchorId, factor.edges.head, factor.prior))
    }
    
    return pose3Graph
  }
  
  /// Solves the unconstrained chordal graph given the original Pose3 graph.
  /// - Parameters:
  ///   - g: The factor graph with only `BetweenFactor<Pose3>` and `PriorFactor<Pose3>`
  ///   - v: the current pose priors
  ///   - ids: the `TypedID`s of the poses
  public func solveOrientationGraph(
    g: FactorGraph,
    v: VariableAssignments,
    ids: Array<TypedID<Pose3>>
  ) -> VariableAssignments {
    /// The orientation graph, with only unconstrained rotation factors
    var orientationGraph = FactorGraph()
    /// orientation storage
    var orientations = VariableAssignments()
    /// association to lookup the vector-based storage from the pose3 ID
    var associations = Dictionary<Int, TypedID<Matrix3>>()
    
    // allocate the space for solved rotations, and memorize the assocation
    for i in ids {
      associations[i.perTypeID] = orientations.store(Matrix3.zero)
    }
    
    // allocate the space for anchor
    associations[anchorId.perTypeID] = orientations.store(Matrix3.zero)
    
    // iterate the pose3 graph and make corresponding relaxed factors
    for factor in g.factors(type: BetweenFactor<Pose3>.self) {
      let R = factor.difference.rot.coordinate.R
      let frob_factor = RelaxedRotationFactorRot3(
        associations[factor.edges.head.perTypeID]!, 
        associations[factor.edges.tail.head.perTypeID]!, R)
      orientationGraph.store(frob_factor)
    }
    
    // make the anchor factor
    orientationGraph.store(RelaxedAnchorFactorRot3(associations[anchorId.perTypeID]!, Matrix3.identity))
    
    // optimize
    var optimizer = GenericCGLS()
    let linearGraph = orientationGraph.linearized(at: orientations)
    optimizer.optimize(gfg: linearGraph, initial: &orientations)
    
    return normalizeRelaxedRotations(orientations, associations: associations, ids: ids)
  }
  
  /// This function finds the closest Rot3 to the unconstrained 3x3 matrix with SVD.
  /// - Parameters:
  ///   - relaxedRot3: the results of the unconstrained chordal optimization
  ///   - associations: mapping from the index of the pose to the index of the corresponding rotation
  ///   - ids: the IDs of the poses
  ///
  /// TODO(fan): replace this with a 3x3 specialized SVD instead of this generic SVD (slow)
  public func normalizeRelaxedRotations(
    _ relaxedRot3: VariableAssignments,
    associations: Dictionary<Int, TypedID<Matrix3>>,
    ids: Array<TypedID<Pose3>>) -> VariableAssignments {
    var validRot3 = VariableAssignments()
    
    for v in ids {
      let M: Matrix3 = relaxedRot3[associations[v.perTypeID]!]
      
      let initRot = Rot3.ClosestTo(mat: M)
      
      // TODO(fan): relies on the assumption of continuous and ordered allocation
      let _ = validRot3.store(initRot)
    }
    
    let M_anchor: Matrix3 = relaxedRot3[associations[anchorId.perTypeID]!]
    
    let initRot_anchor = Rot3.ClosestTo(mat: M_anchor)
    
    // TODO(fan): relies on the assumption of continous and ordered allocation
    let _ = validRot3.store(initRot_anchor)
    
    return validRot3;
  }
  
  /// This function computes the inital poses given the chordal initialized rotations.
  /// - Parameters:
  ///   - graph: The factor graph with only `BetweenFactor<Pose3>` and `PriorFactor<Pose3>`
  ///   - orientations: The orientations returned by the chordal initialization for `Rot3`s
  ///   - ids: the `TypedID`s of the poses
  public func computePoses(graph: FactorGraph, orientations: VariableAssignments, ids: Array<TypedID<Pose3>>) -> VariableAssignments {
    var val = VariableAssignments()
    var g = graph
    for v in ids {
      let _ = val.store(Pose3(orientations[TypedID<Rot3>(v.perTypeID)], Vector3(0,0,0)))
    }
    
    g.store(PriorFactor(anchorId, Pose3()))
    let anchor_id = val.store(Pose3(Rot3(), Vector3(0,0,0)))
    assert(anchor_id == anchorId)
    
    // optimize for 1 G-N iteration
    for _ in 0..<1 {
      let gfg = g.linearized(at: val)
      var dx = val.tangentVectorZeros
      var optimizer = GenericCGLS(precision: 1e-1, max_iteration: 100)
      optimizer.optimize(gfg: gfg, initial: &dx)
      val.move(along: dx)
    }
    return val
  }
  
  /// This function computes the chordal initialization. Normally this is what the user needs to call.
  /// - Parameters:
  ///   - graph: The factor graph with only `BetweenFactor<Pose3>` and `PriorFactor<Pose3>`
  ///   - ids: the `TypedID`s of the poses
  ///
  /// NOTE: This function builds upon the assumption that all variables stored are Pose3s, will fail if that is not the case.
  public static func GetInitializations(graph: FactorGraph, ids: Array<TypedID<Pose3>>) -> VariableAssignments {
    var ci = ChordalInitialization()
    var val = VariableAssignments()
    for _ in ids {
      let _ = val.store(Pose3())
    }
    ci.anchorId = val.store(Pose3())
    // We "extract" the Pose3 subgraph of the original graph: this
    // is done to properly model priors and avoiding operating on a larger graph
    // TODO(fan): This does not work yet as we have not yet reached concensus on how should we
    // handle associations
    let pose3Graph = ci.buildPose3graph(graph: graph)
    
    // Get orientations from relative orientation measurements
    let orientations = ci.solveOrientationGraph(g: pose3Graph, v: val, ids: ids)
    
    // Compute the full poses (1 GN iteration on full poses)
    return ci.computePoses(graph: pose3Graph, orientations: orientations, ids: ids)
  }
}
