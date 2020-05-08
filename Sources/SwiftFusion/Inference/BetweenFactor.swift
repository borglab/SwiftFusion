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

/// A `NonlinearFactor` that calculates the difference of two Values of the same type
///
/// Input is a dictionary of `Key` to `Value` pairs, and the output is the scalar
/// error value
///
/// Interpretation
/// ================
/// `Input`: the input values as key-value pairs
///
public struct BetweenFactor<T: LieGroup>: NonlinearFactor {
  
  var key1: Int
  var key2: Int
  @noDerivative
  public var keys: Array<Int> {
    get {
      [key1, key2]
    }
  }
  public var difference: T
  public typealias Output = Error
  
  public init (_ key1: Int, _ key2: Int, _ difference: T) {
    self.key1 = key1
    self.key2 = key2
    self.difference = difference
  }
  typealias ScalarType = Double
  
  /// TODO: `Dictionary` still does not conform to `Differentiable`
  /// Tracking issue: https://bugs.swift.org/browse/TF-899
//  typealias Input = Dictionary<UInt, Tensor<ScalarType>>

// I want to build a general differentiable dot product
//  @differentiable(wrt: (a, b))
//  static func dot<T: Differentiable & KeyPathIterable>(_ a: T, _ b: T) -> Double {
//    let squared = a.recursivelyAllKeyPaths(to: Double.self).map { a[keyPath: $0] * b[keyPath: $0] }
//
//    return squared.differentiableReduce(0.0, {$0 + $1})
//  }
//
//  @derivative(of: dot)
//  static func _vjpDot<T: Differentiable & KeyPathIterable>(_ a: T, _ b: T) -> (
//    value: Double,
//    pullback: (Double) -> (T.TangentVector, T.TangentVector)
//  ) {
//    return (value: dot(a, b), pullback: { v in
//      ((at.scaled(by: v), bt.scaled(by: v)))
//    })
//  }
  
  /// Returns the `error` of the factor.
  @differentiable(wrt: values)
  public func error(_ values: Values) -> Double {
    let actual = T.differentiableMultiply(values[key1].baseAs(T.self).differentiableInverse(), values[key2].baseAs(T.self))
    let error = difference.differentiableLocal(actual)
    
    return error.differentiableTensor.squared().sum().scalars[0]
  }
  
  @differentiable(wrt: values)
  public func errorVector(_ values: Values) -> T.Coordinate.LocalCoordinate {
    let error = difference.differentiableLocal(
      T.differentiableMultiply(values[key1].baseAs(T.self).differentiableInverse(), values[key2].baseAs(T.self))
    )
    
    return error
  }
  
  public func linearize(_ values: Values) -> JacobianFactor {
    let j = jacobian(of: self.errorVector, at: values)
    
    let j1 = Tensor<Double>(stacking: (0..<j.count).map { i in (j[i]._values[values._indices[key1]!].base as! T.TangentVector).tensor.flattened() })
    let j2 = Tensor<Double>(stacking: (0..<j.count).map { i in (j[i]._values[values._indices[key2]!].base as! T.TangentVector).tensor.flattened() })
    
    let err_tensor = -errorVector(values).tensor
    // TODO: remove this negative sign
    return JacobianFactor(keys, [j1, j2], err_tensor.reshaped(to: [err_tensor.shape[0], 1]))
  }
}
