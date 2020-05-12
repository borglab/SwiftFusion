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

/// The objective function of a linear least squares optimization problem.
///
/// The problem is to find the value `x: Variables` that minimizes `energy(at: x)`, where
/// `energy(at: x)` is defined to be the Euclidean norm of `residuals(at: x)`.
public protocol LinearLeastSquaresObjective {
  /// The type of the solution.
  associatedtype Variables: EuclideanVectorSpace

  /// The type of the residual.
  associatedtype Residuals: EuclideanVectorSpace

  /// An affine function of `x`.
  func residuals(at x: Variables) -> Residuals
}

extension LinearLeastSquaresObjective {
  /// The objective function that we are trying to minimize.
  func energy(at x: Variables) -> Residuals.VectorSpaceScalar {
    return residuals(at: x).squaredNorm
  }
}

/// The objective function of a linear least squares optimization problem whose residuals are
/// given by `bias - A * x`, where `A` is a matrix and `bias` is a vector.
public protocol MatrixLinearLeastSquaresObjective: LinearLeastSquaresObjective {
  /// The bias term.
  var bias: Residuals { get }

  /// Returns the product `A * x`.
  func productA(times x: Variables) -> Residuals

  /// Returns the product `A^t * r`.
  func productATranspose(times r: Residuals) -> Variables
}

extension MatrixLinearLeastSquaresObjective {
  /// The residuals are determined by the matrix and bias, so a conforming type does not have to
  /// define `residuals` itself.
  public func residuals(at x: Variables) -> Residuals {
    return bias - productA(times: x)
  }
}
