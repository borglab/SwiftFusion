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

public class CGLS<Model: Differentiable>
  where Model.TangentVector: TangentStandardBasis & VectorProtocol & ElementaryFunctions,
  Model.TangentVector.VectorSpaceScalar == Double {
  public typealias Model = Model
  /// The set of steps taken.
  public var step: Int = 0

  public init(
    for _: __shared Model) {
    
  }

  public func optimize(loss f: @differentiable (Model) -> Model.TangentVector.VectorSpaceScalar, model x_in: inout Model) {
    step += 1
    
    let x_0 = x_in
    
    let dx_0 = gradient(at: x_0, in: f)
    
    let a_0 = 1.0
    // a_0 = argmin(f(x_0+a*dx_0))
    
    var x_1 = x_0
    x_1.move(dx_0.scaled(by: a_0))
    
    var x_n = x_1
    var dx_n_1 = dx_0
    var s = dx_0
    
    while true {
      let dx = gradient(at: x_n, in: f)
      
      // Fletcher-Reeves
      let beta = dx.dot(dx) / dx_n_1.dot(dx_n_1)
      
      // s_n = \delta x_n + \beta_n s_{n-1}
      s = dx + beta * s
      
      // Line search
      // let a = argmin(f(x_n + a * s))
      let a = 1.0
      
      x_n = x_n + s.scaled(by: a)
    }
    
    model.move(along: A_T)
  }
}
