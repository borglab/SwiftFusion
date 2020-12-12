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

/// A BetweenFactor alternative that uses the Chordal (Frobenious) norm on rotation for Pose3
public struct BetweenFactorAlternative: LinearizableFactor2 {
  public let edges: Variables.Indices
  public let difference: Pose3

  public init(_ startId: TypedID<Pose3>, _ endId: TypedID<Pose3>, _ difference: Pose3) {
    self.edges = Tuple2(startId, endId)
    self.difference = difference
  }

  @differentiable
  public func errorVector(_ start: Pose3, _ end: Pose3) -> Vector12 {
    let actualMotion = between(start, end)
    let R = actualMotion.coordinate.rot.coordinate.R + (-1) * difference.rot.coordinate.R
    let t = actualMotion.t - difference.t
    
    return Vector12(concatenating: R, t)
  }
}
