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
///
/// Note: This is currently named with a "Generic" prefix to avoid clashing with the other factors.
/// When we completely replace the existing factors with the "Generic" ones, we should remove this
/// prefix.
struct GenericPriorFactor<Pose: LieGroup, JacobianRows: FixedSizeArray>:
  GenericLinearizableFactor
  where JacobianRows.Element == Tuple1<Pose.TangentVector>
{
  typealias Variables = Tuple1<Pose>

  let edges: Variables.Indices
  let prior: Pose

  typealias ErrorVector = Pose.TangentVector
  func errorVector(_ x: Pose) -> ErrorVector {
    return prior.localCoordinate(x)
  }

  // Note: All the remaining code in this factor is boilerplate that we can eventually eliminate
  // with sugar.
  func error(at x: Variables) -> Double {
    return errorVector(at: x).squaredNorm
  }

  func errorVector(at x: Variables) -> Pose.TangentVector {
    return errorVector(x.head)
  }

  typealias Linearized = GenericJacobianFactor<JacobianRows, ErrorVector>
  func linearized(at x: Variables) -> Linearized {
    Linearized(linearizing: errorVector, at: x, edges: edges)
  }
}

typealias GenericPriorFactor2 = GenericPriorFactor<Pose2, Array3<Tuple1<Vector3>>>
