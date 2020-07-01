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

/// A Gaussian distribution over the input, represented as a materialized Jacobian matrix and
/// materialized error vector.
public struct JacobianFactor<
  Rows: FixedSizeArray,
  ErrorVector: EuclideanVectorN
>: LinearApproximationFactor where Rows.Element: EuclideanVectorN & DifferentiableVariableTuple {
  public typealias Variables = Rows.Element

  /// The Jacobian matrix, as a fixed size array of rows.
  ///
  /// The `jacobian` has one row per element of the `ErrorVector`, and each row is a vector in the
  /// vector space of adjacent variables. For example, if `ErrorVector == Vector3` and
  /// `Variables == Tuple2<Vector3, Vector3>`, then `Rows == Array3<Tuple2<Vector3, Vector3>>`. See
  /// the typealiases below for more examples.
  public let jacobian: Rows

  /// The error vector.
  public let error: ErrorVector

  /// The ids of the variables adjacent to this factor.
  public let edges: Variables.Indices

  /// Creates a Jacobian factor with the given `jacobian`, `error`, and `edges`.
  public init(jacobian: Rows, error: ErrorVector, edges: Variables.Indices) {
    self.jacobian = jacobian
    self.error = error
    self.edges = edges
  }

  /// Creates a factor that linearly approximates `f` at `x`.
  public init<F: LinearizableFactor>(linearizing f: F, at x: F.Variables)
  where F.Variables.TangentVector == Variables, F.ErrorVector == ErrorVector {
    let (value, pb) = valueWithPullback(at: x, in: f.errorVector)
    let rows = Rows(ErrorVector.standardBasis.lazy.map(pb))
    self.jacobian = rows
    self.error = value
    self.edges = F.Variables.linearized(f.edges)
  }

  public func error(at x: Variables) -> Double {
    return errorVector(at: x).squaredNorm
  }

  @differentiable
  public func errorVector(at x: Variables) -> ErrorVector {
    return error - errorVector_linearComponent(x)
  }

  @usableFromInline
  @derivative(of: errorVector)
  func vjpErrorVector(at x: Variables) -> (value: ErrorVector, pullback: (ErrorVector) -> Variables) {
    return (errorVector(at: x), errorVector_linearComponent_adjoint)
  }

  public func errorVector_linearComponent(_ x: Variables) -> ErrorVector {
    // The compiler isn't able to optimize the closure away if we map `jacobian`, but it is able
    // to optimize the closure away if we map `jacobian`'s `UnsafeBufferPointer`.
    jacobian.withUnsafeBufferPointer { rows in
      ErrorVector(rows.lazy.map { $0.dot(x) })
    }
  }

  public func errorVector_linearComponent_adjoint(_ y: ErrorVector) -> Variables {
    // We use `UnsafeBufferPointer`s to avoid forming collections that can't be optimized away.
    y.withUnsafeBufferPointer { scalars in
      jacobian.withUnsafeBufferPointer { rows in
        // We reduce the range `0..<ErrorVector.dimension` instead of `zip(scalars, rows)`, to
        // avoid forming collections that can't be optimized away.
        // TODO: This is not getting unrolled after `ErrorVector` is specialized. Convincing the
        // optimizer to unroll it might speed things up.
        (0..<ErrorVector.dimension).reduce(into: Variables.zero) { (result, i) in
          result += scalars[i] * rows[i]
        }
      }
    }
  }
}

/// A Jacobian factor with 1 3-dimensional input and a 3-dimensional error vector.
public typealias JacobianFactor3x3_1 = JacobianFactor<Array3<Tuple1<Vector3>>, Vector3>

/// A Jacobian factor with 2 3-dimensional inputs and a 3-dimensional error vector.
public typealias JacobianFactor3x3_2 = JacobianFactor<Array3<Tuple2<Vector3, Vector3>>, Vector3>

/// A Jacobian factor with 1 6-dimensional input and a 6-dimensional error vector.
public typealias JacobianFactor6x6_1 = JacobianFactor<Array6<Tuple1<Vector6>>, Vector6>

/// A Jacobian factor with 2 6-dimensional inputs and a 6-dimensional error vector.
public typealias JacobianFactor6x6_2 = JacobianFactor<Array6<Tuple2<Vector6, Vector6>>, Vector6>
