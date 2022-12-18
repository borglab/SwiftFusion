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
import PenguinStructures

/// A factor that specifies a difference between two instances of a Group.
public struct BetweenFactor<Group: LieGroup>: LinearizableFactor2 {
  public let edges: Variables.Indices
  public let difference: Group
  
  public init(_ startId: TypedID<Group>, _ endId: TypedID<Group>, _ difference: Group) {
    self.edges = Tuple2(startId, endId)
    self.difference = difference
  }
  
  @differentiable(reverse)
  public func errorVector(_ start: Group, _ end: Group) -> Group.TangentVector {
    let actualMotion = between(start, end)
    return difference.localCoordinate(actualMotion)
  }
}

/// A factor that specifies a difference between two instances of a Group, version with weight.
public struct WeightedBetweenFactor<Group: LieGroup>: LinearizableFactor2 {
  public let edges: Variables.Indices
  public let difference: Group
  public let weight: Double
  
  public init(_ startId: TypedID<Group>, _ endId: TypedID<Group>, _ difference: Group, weight: Double) {
    self.edges = Tuple2(startId, endId)
    self.difference = difference
    self.weight = weight
  }
  
  @differentiable(reverse)
  public func errorVector(_ start: Group, _ end: Group) -> Group.TangentVector {
    let actualMotion = between(start, end)
    return weight * difference.localCoordinate(actualMotion)
  }
}
