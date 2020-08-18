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
public struct SwitchingBetweenFactor<Pose: LieGroup>: VectorFactor3 {
  public typealias ErrorVector = Pose.TangentVector
  public typealias LinearizableComponent = BetweenFactor<Pose>

  public let edges: Variables.Indices

  /// Movement templates for each label.
  let motions: [Pose]
  
  public init(_ from: TypedID<Pose>,
              _ label: TypedID<Int>,
              _ to: TypedID<Pose>,
              _ motions: [Pose]) {
    self.edges = Tuple3(from, label, to)
    self.motions = motions
  }
  
  public func errorVector(_ from: Pose, _ motionLabel: Int, _ to: Pose) -> ErrorVector {
    let actualMotion = between(from, to)
    return motions[motionLabel].localCoordinate(actualMotion)
  }

  public func linearizableComponent(_ from: Pose, _ motionLabel: Int, _ to: Pose)
    -> (LinearizableComponent, LinearizableComponent.Variables)
  {
    return (
      BetweenFactor(input0ID, input2ID, motions[motionLabel]),
      Tuple2(from, to)
    )
  }
}

