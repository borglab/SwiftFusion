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
import TensorFlow
import PenguinStructures

/// A `NonlinearFactor` that calculates the bearing and range error of one pose and one landmark
///
public struct BearingRangeFactor2 : LinearizableFactor2 {
  public typealias Base = Pose2
  public typealias Target = Vector2
  public typealias Bearing = Rot2

  public let edges: Variables.Indices
  public let bearingMeas: Bearing
  public let rangeMeas: Double


  public init(_ baseId: TypedID<Base>, _ targetId: TypedID<Target>, _ bearingMeas: Bearing, _ rangeMeas: Double) {
    self.edges = Tuple2(baseId, targetId)
    self.bearingMeas = bearingMeas
    self.rangeMeas = rangeMeas
  }

  public typealias Variables = Tuple2<Base, Target>
  @differentiable
  public func errorVector(_ base: Base, _ target: Target) -> Vector2 {
    let dx = (target - base.t)
    let actual_bearing = between(Rot2(c: dx.x / dx.norm, s: dx.y / dx.norm), base.rot)
    let actual_range = dx.norm
    let error_range = (actual_range - rangeMeas)
    let error_bearing = between(actual_bearing, bearingMeas)
    
    return Vector2(error_bearing.theta, error_range)
  }
}
