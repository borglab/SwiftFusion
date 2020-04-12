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
/// Non-Linear Conjugate Gradient (NLCG) optimizer.
///
/// An optimizer that implements NLCG second order optimizer

public class NLCG<Model: Differentiable>
  where Model.TangentVector: VectorProtocol & ElementaryFunctions & KeyPathIterable,
  Model.TangentVector.VectorSpaceScalar == Double {
  public typealias Model = Model
  /// The set of steps taken.
  public var step: Int = 0
  public var precision: Double = 1e-10
  public var max_iteration: Int = 400
  
  public init(
    for _: __shared Model, precision p: Double = 1e-10, max_iteration maxiter: Int = 400) {
    
    precision = p
    max_iteration = maxiter
  }

  func dot<T: Differentiable>(_ for: T, _ a: T.TangentVector, _ b: T.TangentVector) -> Double where T.TangentVector: KeyPathIterable {
    a.recursivelyAllWritableKeyPaths(to: Double.self).map { a[keyPath: $0] * b[keyPath: $0] }.reduce(0.0, {$0 + $1})
  }
  
  public func optimize(loss f: @differentiable (Model) -> Model.TangentVector.VectorSpaceScalar, model x_in: inout Model) {
    step = 0
    
    let x_0 = x_in
    
    let dx_0 = gradient(at: x_0, in: f)
    
    let a_0 = 1.0
    // a_0 = argmin(f(x_0+a*dx_0))
    
    var x_1 = x_0
    x_1.move(along: dx_0.scaled(by: a_0))
    
    var x_n = x_1
    var dx_n_1 = dx_0
    var s = dx_0
    
    while step < max_iteration {
      let dx = gradient(at: x_n, in: f)
      
      // Fletcher-Reeves
      // TODO: `.dot` needs to be implemented by iterating over keyPath
      let beta: Double = dot(x_in, dx, dx) / dot(x_in, dx_n_1, dx_n_1)
      
      // s_n = \delta x_n + \beta_n s_{n-1}
      s = dx + s.scaled(by: beta)
      
      debugPrint("dx = ", dx)
      debugPrint("s = ", s)
      debugPrint("x_n = ", x_n)
      // Line search
      // let a = argmin(f(x_n + a * s))
      let f_a: @differentiable (_ a: Double) -> Model.TangentVector.VectorSpaceScalar = { a in
        var x = x_n
        x.move(along: s.scaled(by: a))
        return f(x)
      }
      var a = 1.0
      
      let sgd = SGD(for: a)
      for _ in 0..<100 {
        let 𝛁loss = gradient(at: a, in: f_a)
        sgd.update(&a, along: 𝛁loss)
      }

      x_n.move(along: s.scaled(by: a))
      dx_n_1 = dx
      step += 1
    }
    
  }
}
