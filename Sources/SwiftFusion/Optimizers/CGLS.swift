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
/// Conjugate Gradient Least Squares (CGLS) optimizer.
///
/// An optimizer that implements CGLS second order optimizer
public class CGLS {
  /// The set of steps taken.
  public var step: Int = 0

  public init() {}

  public func optimize(gfg: GaussianFactorGraph, initial: Values) {
    step += 1
    
    let b_0 = gfg.b
    let A = gfg.A
    var x = initial
    var r = b - A*x
    var p = A.T * r
    var s = p
    var gamma = s.norm
    
    while true {
      let q = A * p
      let alpha = gamma / q.norm
      x = x + alpha * p
      r = r - alpha * q
      s = A.T * r
      
      let gamma_next = s.norm
      let beta = gamma_next/gamma
      gamma = gamma_next
      p = s + beta * p
    }
    
    model.move(along: A_T)
  }
}
