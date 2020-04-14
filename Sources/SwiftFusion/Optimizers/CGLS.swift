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

  /// Constructor (Probably not necessary)
  public init() {}

  /// Optimize the Gaussian Factor Graph with a initial estimate
  /// Reference: Bjorck96book_numerical-methods-for-least-squares-problems
  /// Page 289, Algorithm 7.4.1
  public func optimize(gfg: GaussianFactorGraph, initial: VectorValues) {
    step += 1
    
    let b = gfg.b
    
    var x: VectorValues = initial // x(0)
    var r: Errors = b - gfg * x // r(0) = b - A * x(0)
    var p = gfg.atr(r: r) // p(0) = s(0) = A^T * r(0)
    var s = p
    var gamma = s.norm // γ(0) = ||s(0)||^2
    
    while true {
      let q = gfg * p // q(k) = A * p(k)
      let alpha: Double = gamma / q.norm // α(k) = γ(k)/||q(k)||^2
      x = x + (alpha * p) // x(k+1) = x(k) + α(k) * p(k)
      r = r + (-alpha) * q // r(k+1) = r(k) - α(k) * q(k)
      s = gfg.atr(r: r) // s(k+1) = A.T * r(k+1)
      
      let gamma_next = s.norm // γ(k+1) = ||s(k+1)||^2
      let beta: Double = gamma_next/gamma // β(k) = γ(k+1)/γ(k)
      gamma = gamma_next
      p = s + beta * p // p(k+1) = s(k+1) + β(k) * p(k)
    }
  }
}
