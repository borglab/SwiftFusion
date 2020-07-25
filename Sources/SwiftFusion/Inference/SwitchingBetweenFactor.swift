//
//  SwitchingBetweenFactor.swift
//  
//
//  Frank Dellaert and Marc Rasi
//  July 2020

import Foundation
import PenguinStructures

/// A factor with a switchable motion model.
///
/// `JacobianRows` specifies the `Rows` parameter of the Jacobian of this factor. See the
/// documentation on `JacobianFactor.jacobian` for more information. Use the typealiases below to
/// avoid specifying this type parameter every time you create an instance.
public struct SwitchingBetweenFactor<Pose: LieGroup, JacobianRows: FixedSizeArray>:
  LinearizableFactor
where JacobianRows.Element == Tuple2<Pose.TangentVector, Pose.TangentVector>
{
  public typealias Variables = Tuple3<Pose, Int, Pose>
  
  public let edges: Variables.Indices
  
  /// Movement temmplates for each label.
  let movements: [Pose]
  
  public init(_ from: TypedID<Pose>,
              _ label: TypedID<Int>,
              _ to: TypedID<Pose>,
              _ movements: [Pose]) {
    self.edges = Tuple3(from, label, to)
    self.movements = movements
  }
  
  public typealias ErrorVector = Pose.TangentVector
  
  @differentiable(wrt: (start, end))
  public func errorVector(_ start: Pose, _ label: Int, _ end: Pose) -> ErrorVector {
    let actualMotion = between(start, end)
    return movements[label].localCoordinate(actualMotion)
  }
  
  // Note: All the remaining code in this factor is boilerplate that we can eventually eliminate
  // with sugar.
  
  public func error(at x: Variables) -> Double {
    return errorVector(at: x).squaredNorm
  }
  
  public func errorVector(at x: Variables) -> Pose.TangentVector {
    return errorVector(x.head, x.tail.head, x.tail.tail.head)
  }
  
  public typealias Linearization = JacobianFactor<JacobianRows, ErrorVector>
  public func linearized(at x: Variables) -> Linearization {
    let (start, label, end) = (x.head, x.tail.head, x.tail.tail.head)
    let (startEdge, endEdge) = (edges.head, edges.tail.tail.head)
    let differentiableVariables = Tuple2(start, end)
    let differentiableEdges = Tuple2(startEdge, endEdge)
    return Linearization(
      linearizing: { differentiableVariables in
        let (start, end) = (differentiableVariables.head, differentiableVariables.tail.head)
        return errorVector(start, label, end)
      },
      at: differentiableVariables,
      edges: differentiableEdges
    )
  }
}

