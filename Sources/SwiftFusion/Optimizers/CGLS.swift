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
import TensorFlow

/// Conjugate Gradient Least Squares (CGLS) optimizer.
///
/// An optimizer that implements CGLS second order optimizer
public class CGLS {
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
  public func optimize<Objective: MatrixLinearLeastSquaresObjective>(
    objective: Objective,
    initial: inout Objective.Variables
  ) {
    step += 1

    var x: Objective.Variables = initial // x(0), the initial value
    var r: Objective.Residuals = objective.residuals(at: x) // r(0) = b - A * x(0), the residual
    var p = objective.productATranspose(times: r) // p(0) = s(0) = A^T * r(0), residual in value space
    var s = p // residual of normal equations
    var gamma = s.squaredNorm // γ(0) = ||s(0)||^2
    
    while step < max_iteration {
      let q = objective.productA(times: p) // q(k) = A * p(k)
      let alpha: Double = gamma / q.squaredNorm // α(k) = γ(k)/||q(k)||^2
      x = x + p.scaled(by: alpha) // x(k+1) = x(k) + α(k) * p(k)
      r = r + q.scaled(by: -alpha) // r(k+1) = r(k) - α(k) * q(k)
      s = objective.productATranspose(times: r) // s(k+1) = A.T * r(k+1)
      
      let gamma_next = s.squaredNorm // γ(k+1) = ||s(k+1)||^2
      let beta: Double = gamma_next/gamma // β(k) = γ(k+1)/γ(k)
      gamma = gamma_next
      p = s + p.scaled(by: beta) // p(k+1) = s(k+1) + β(k) * p(k)
      
      if alpha * alpha * p.squaredNorm < precision {
        break
      }
      step += 1
    }
    
    initial = x
  }
}
