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

/// A Gaussian factor stored as a jacobian matrix and an error vector.
///
/// Note: This is currently named with a "Generic" prefix to avoid clashing with the other factors.
/// When we completely replace the existing factors with the "Generic" ones, we should remove this
/// prefix.
struct GenericJacobianFactor<
  Rows: FixedSizeArray,
  ErrorVector: EuclideanVectorN
>: GenericGaussianFactor where Rows.Element: EuclideanVectorN & VariableTuple {
  typealias Variables = Rows.Element

  /// The Jacobian matrix, as a fixed size array of rows.
  let jacobian: Rows

  /// The error vector.
  let error: ErrorVector

  /// The ids of the variables adjacent to this factor.
  let edges: Variables.Indices

  /// Creates a Jacobian factor with the given `jacobian`, `error`, and `edges`.
  init(jacobian: Rows, error: ErrorVector, edges: Variables.Indices) {
    self.jacobian = jacobian
    self.error = error
    self.edges = edges
  }

  /// Creates a Jacobian factor that linearizes `f` at `x`, and is adjacent to the variables
  /// identifed by edges.
  init<Input: Differentiable & DifferentiableVariableTuple>(
    linearizing f: @differentiable (Input) -> ErrorVector,
    at x: Input,
    edges: Input.Indices
  ) where Input.TangentVector == Variables, Input.TangentIndices == Variables.Indices {
    let (value, pb) = valueWithPullback(at: x, in: f)
    let rows = Rows(ErrorVector.standardBasis.lazy.map(pb))
    self.jacobian = rows
    self.error = value
    self.edges = Input.tangentIndices(edges)
  }

  func error(at x: Variables) -> Double {
    return errorVector(at: x).squaredNorm
  }

  func errorVector(at x: Variables) -> ErrorVector {
    return linearForward(x) + error
  }

  func linearForward(_ x: Variables) -> ErrorVector {
    return ErrorVector(jacobian.lazy.map { $0.dot(x) })
  }

  func linearAdjoint(_ y: ErrorVector) -> Variables {
    return zip(y.scalars, jacobian).lazy.map(*).reduce(Variables.zero, +)
  }

  typealias Linearized = Self
  func linearized(at x: Variables) -> Self {
    return self
  }
}

typealias JacobianFactor3x3 = GenericJacobianFactor<Array3<Tuple1<Vector3>>, Vector3>
typealias JacobianFactor3x3_2 = GenericJacobianFactor<Array3<Tuple2<Vector3, Vector3>>, Vector3>
