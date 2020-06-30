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

/// Non-Linear Conjugate Gradient (NLCG) optimizer.
///
/// An optimizer that implements NLCG second order optimizer
/// It is generic over all differentiable models that is `KeyPathIterable`
/// This loosely follows `Nocedal06book_numericalOptimization`, page 121
public class NLCG<Model: Differentiable & KeyPathIterable>
  where Model.TangentVector: EuclideanVector & KeyPathIterable
{
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
  
  /// dot product
  /// TODO: this needs to be moved to Core
  func dot<T: Differentiable>(_ for: T, _ a: T.TangentVector, _ b: T.TangentVector) -> Double where T.TangentVector: KeyPathIterable {
    a.recursivelyAllWritableKeyPaths(to: Double.self).map { a[keyPath: $0] * b[keyPath: $0] }.reduce(0.0, {$0 + $1})
  }
  
  func lineSearch(f: @differentiable @escaping (Model) -> Double,
                  currentValues: Model, gradient: Model.TangentVector) -> Double {
    /* normalize it such that it becomes a unit vector */
    let g = sqrt(dot(currentValues, gradient, gradient))

    // perform the golden section search algorithm to decide the the optimal step size
    // detail refer to http://en.wikipedia.org/wiki/Golden_section_search
    let phi: Double = 0.5 * (1.0 + sqrt(5.0))
    let resphi: Double = 2.0 - phi
    let tau: Double = 1e-5
    
    var minStep: Double = -1.0 / g
    var maxStep: Double = 0.0
    var newStep: Double = minStep + (maxStep - minStep) / (phi + 1.0)

    var newValues = currentValues
    newValues.move(along: newStep * gradient)
    var newError = f(newValues);

    while (true) {
      let flag: Bool = (maxStep - newStep > newStep - minStep) ? true : false;
      let testStep =
          flag ? newStep + resphi * (maxStep - newStep) :
              newStep - resphi * (newStep - minStep);

      if ((maxStep - minStep) < tau * (abs(testStep) + abs(newStep))) {
        return 0.5 * (minStep + maxStep);
      }

      var testValues = currentValues
      testValues.move(along: testStep * gradient)
      let testError = f(testValues)

      // update the working range
      if (testError >= newError) {
        if (flag) {
          maxStep = testStep
        } else {
          minStep = testStep
        }
      } else {
        if (flag) {
          minStep = newStep;
          newStep = testStep;
          newError = testError;
        } else {
          maxStep = newStep;
          newStep = testStep;
          newError = testError;
        }
      }
    }
  }
  
  /// Optimize with an initial estimate `model`
  public func optimize(loss f: @differentiable @escaping (Model) -> Double, model x_in: inout Model) {
    step = 0
    
    let x_0 = x_in
    
    let dx_0 = gradient(at: x_0, in: f)
    
    // a_0 = argmin(f(x_0+a*dx_0))
    let a_0 = lineSearch(f: f, currentValues: x_0, gradient: dx_0)
    
    var x_1 = x_0
    x_1.move(along: a_0 * dx_0)
    
    var x_n = x_1
    var dx_n_1 = dx_0
    var s = dx_0
    
    while step < max_iteration {
      let dx = gradient(at: x_n, in: f)
      
      // Fletcher-Reeves
      // TODO: `.dot` needs to be implemented by iterating over keyPath
      let beta: Double = dot(x_in, dx, dx) / dot(x_in, dx_n_1, dx_n_1)
      
      // s_n = \delta x_n + \beta_n s_{n-1}
      s = dx + beta * s
      
      // Line search
      let a = lineSearch(f: f, currentValues: x_n, gradient: s)

      let delta = a * s
      
      x_n.move(along: delta) // update the estimate
      
      // Exit when delta is too small
      if dot(x_n, delta, delta) < precision {
        break
      }
      
      dx_n_1 = dx
      step += 1
    }
    
    // Finally, assign back to the value passed in
    x_in = x_n
  }
}
