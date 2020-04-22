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
public struct BetweenFactor: NonlinearFactor {
  @noDerivative
  public var keys: Array<Int> = []
  public var difference: Pose2
  public typealias Output = Error
  
  public init (_ key1: Int, _ key2: Int, _ difference: Pose2) {
    keys = [key1, key2]
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
    let error = between(
      between(values[keys[1]].baseAs(Pose2.self), values[keys[0]].baseAs(Pose2.self)),
        difference
    )
    
    return error.t.norm + error.rot.theta * error.rot.theta
  }
  
  @differentiable(wrt: values)
  public func errorVector(_ values: Values) -> Vector3 {
    let error = between(
      between(values[keys[1]].baseAs(Pose2.self), values[keys[0]].baseAs(Pose2.self)),
      difference
    )
    
    return Vector3(error.rot.theta, error.t.x, error.t.y)
  }
  
  public func linearize(_ values: Values) -> JacobianFactor {
    let j = jacobian(of: self.errorVector, at: values)
    
    let j1 = Tensor<Double>(stacking: (0..<3).map { i in (j[i]._values[values._indices[keys[0]]!].base as! Pose2.TangentVector).tensor.reshaped(to: TensorShape([3])) })
    let j2 = Tensor<Double>(stacking: (0..<3).map { i in (j[i]._values[values._indices[keys[1]]!].base as! Pose2.TangentVector).tensor.reshaped(to: TensorShape([3])) })
    
    // TODO: remove this negative sign
    return JacobianFactor(keys, [j1, j2], -errorVector(values).tensor.reshaped(to: [3, 1]))
  }
}
