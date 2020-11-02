// Copyright 2019 The SwiftFusion Authors. All Rights Reserved.
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

/// Conjugate Gradient Least Squares (CGLS) optimizer.
///
/// An optimizer that implements CGLS second order optimizer
public struct GenericCGLS {
  /// The set of steps taken.
  public var step: Int = 0
  public var precision: Double = 1e-10
  public var max_iteration: Int = 400

  /// Constructor
  public init(precision p: Double = 1e-10, max_iteration maxiter: Int = 400) {
    precision = p
    max_iteration = maxiter
  }

  /// Optimize the Gaussian Factor Graph with a initial estimate
  /// Reference: Bjorck96book_numerical-methods-for-least-squares-problems
  /// Page 289, Algorithm 7.4.1
  public mutating func optimize(gfg: GaussianFactorGraph, initial x: inout VariableAssignments) {
    startTimer("cgls")
    defer { stopTimer("cgls") }
    step += 1

    var r = (-1) * gfg.errorVectors(at: x) // r(0) = b - A * x(0), the residual
    var p = gfg.errorVectors_linearComponent_adjoint(r) // p(0) = s(0) = A^T * r(0), residual in value space
    var s = p // residual of normal equations
    var gamma = s.squaredNorm // γ(0) = ||s(0)||^2

    while step < max_iteration && gamma > precision {
      incrementCounter("cgls step")
      // print("[CGLS    ] residual = \(r.squaredNorm), true = \(gfg.errorVectors(at: x).squaredNorm)")
      let q = gfg.errorVectors_linearComponent(at: p) // q(k) = A * p(k)

      let alpha: Double = gamma / q.squaredNorm // α(k) = γ(k)/||q(k)||^2
      x = x + (alpha * p) // x(k+1) = x(k) + α(k) * p(k)
      r = r + (-alpha) * q // r(k+1) = r(k) - α(k) * q(k)
      s = gfg.errorVectors_linearComponent_adjoint(r) // s(k+1) = A.T * r(k+1)

      let gamma_next = s.squaredNorm // γ(k+1) = ||s(k+1)||^2
      let beta: Double = gamma_next/gamma // β(k) = γ(k+1)/γ(k)
      gamma = gamma_next
      p = s + beta * p // p(k+1) = s(k+1) + β(k) * p(k)

      if (alpha * p).squaredNorm < precision {
        break
      }
      step += 1
    }
  }
}
