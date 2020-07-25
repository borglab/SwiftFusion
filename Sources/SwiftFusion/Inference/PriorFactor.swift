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

/// A factor that specifies a prior on a pose.
public struct PriorFactor<Pose: LieGroup>: LinearizableFactor1 {
  public let edges: Variables.Indices
  public let prior: Pose

  public init(_ id: TypedID<Pose>, _ prior: Pose) {
    self.edges = Tuple1(id)
    self.prior = prior
  }

  @differentiable
  public func errorVector(_ x: Pose) -> Pose.TangentVector {
    return prior.localCoordinate(x)
  }
}
