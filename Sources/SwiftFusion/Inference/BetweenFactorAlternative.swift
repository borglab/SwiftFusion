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

fileprivate extension Vector12 {
  @differentiable
  init(concatenating m: Matrix3, _ v: Vector3) {
    self.s0 = m[0, 0]
    self.s1 = m[0, 1]
    self.s2 = m[0, 2]
    self.s3 = m[1, 0]
    self.s4 = m[1, 1]
    self.s5 = m[1, 2]
    self.s6 = m[2, 0]
    self.s7 = m[2, 1]
    self.s8 = m[2, 2]
    self.s9 = v.x
    self.s10 = v.y
    self.s11 = v.z
  }
}
