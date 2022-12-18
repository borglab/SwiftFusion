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

/// A linear approximation of a `GaussianFactor`.
///
/// Since `GaussianFactor`s are linear, they are their own linear approximations.
public struct IdentityLinearizationFactor<Base: GaussianFactor>: LinearApproximationFactor {
  /// The appoximated factor.
  let base: Base

  /// A tuple of the variable types of variables adjacent to this factor.
  public typealias Variables = Base.Variables

  /// The type of the error vector.
  public typealias ErrorVector = Base.ErrorVector

  /// The IDs of the variables adjacent to this factor.
  public var edges: Variables.Indices {
    base.edges
  }

  /// Creates a factor that linearly approximates `f` at `x`.
  ///
  /// - Requires: `F == Base`.
  public init<F: LinearizableFactor>(linearizing f: F, at x: F.Variables)
  where F.Variables.TangentVector == Variables, F.ErrorVector == ErrorVector {
    self.base = f as! Base
  }

  /// Returns the error at `x`.
  ///
  /// This is typically interpreted as negative log-likelihood.
  public func error(at x: Variables) -> Double {
    base.error(at: x)
  }

  /// Returns the error vector given the values of the adjacent variables.
  @differentiable(reverse)
  public func errorVector(at x: Variables) -> ErrorVector {
    base.errorVector(at: x)
  }

  /// The linear component of `errorVector`.
  public func errorVector_linearComponent(_ x: Variables) -> ErrorVector {
    base.errorVector_linearComponent(x)
  }

  /// The adjoint (aka "transpose" or "dual") of the linear component of `errorVector`.
  public func errorVector_linearComponent_adjoint(_ y: ErrorVector) -> Variables {
    base.errorVector_linearComponent_adjoint(y)
  }
}
