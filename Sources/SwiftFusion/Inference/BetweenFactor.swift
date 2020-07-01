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

/// A factor that specifies a difference between two poses.
public struct BetweenFactor<Pose: LieGroup>: LinearizableFactor {
  public typealias Variables = Tuple2<Pose, Pose>

  public let edges: Variables.Indices
  public let difference: Pose

  public init(_ startId: TypedID<Pose, Int>, _ endId: TypedID<Pose, Int>, _ difference: Pose) {
    self.edges = Tuple2(startId, endId)
    self.difference = difference
  }

  public typealias ErrorVector = Pose.TangentVector
  public func errorVector(_ start: Pose, _ end: Pose) -> ErrorVector {
    let actualMotion = between(start, end)
    return difference.localCoordinate(actualMotion)
  }

  // Note: All the remaining code in this factor is boilerplate that we can eventually eliminate
  // with sugar.
  
  public func error(at x: Variables) -> Double {
    return errorVector(at: x).squaredNorm
  }

  @differentiable
  public func errorVector(at x: Variables) -> Pose.TangentVector {
    return errorVector(x.head, x.tail.head)
  }
}
