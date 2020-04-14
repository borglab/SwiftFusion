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

/// The type of error vector returned by the Factor
/// Should be VectorN, but now just tensor as we do not have fixed size Tensors
public typealias Error = Tensor<Double>

/// Collection of all errors returned by a Factor Graph
public typealias Errors = Array<Error>

/// Extending Array for Error type
extension Array where Element == Error {
  public static func - (_ a: Self, _ b: Self) -> Self {
    var result = a
    let _ = result.indices.map { result[$0] = a[$0] + b[$0] }
    return result
  }
  
  /// Calculates the L2 norm
  public var norm: Double {
    get {
      self.map { $0.squared().sum().scalar! }.reduce(0.0, { $0 + $1 })
    }
  }
  
  /// Errors + scalar
  static func + (_ lhs: Self, _ rhs: Double) -> Self {
    var result = lhs
    let _ = result.indices.map { result[$0] += rhs }
    return result
  }
  
  /// Errors + Errors
  static func + (_ lhs: Self, _ rhs: Self) -> Self {
    var result = lhs
    let _ = result.indices.map { result[$0] += rhs[$0] }
    return result
  }
  
  /// scalar * Errors
  static func * (_ lhs: Double, _ rhs: Self) -> Self {
    var result = rhs
    let _ = result.indices.map { result[$0] *= lhs }
    return result
  }
}
