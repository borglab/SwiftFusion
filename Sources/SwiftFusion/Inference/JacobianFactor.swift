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

/// A `LinearFactor` that operates like `JacobianFactor` in GTSAM.
///
/// Input is a dictionary of `Key` to `Tensor` pairs, and the output is the paired
/// error vector. Note here that the Tensor shapes are not checked.
///
/// Interpretation
/// ================
/// `Input`: the input values as key-value pairs
/// `Errors`: the vector `J * x`, where `J` is `m*n` and `x` is `n*1` and returns a m vector
///
/// Explanation
/// ================
/// TODO:
/// I think both Jacobian and Hessian factors in GTSAM are converted into JacobianFactor in
/// `GaussianFactorGraph`, so it becomes a question whether we should do the same?
/// I am considering making `JacobianLikeFactor` a protocol and make `JacobianFactor`
/// and `HessianFactor` conform to this protocol instead.
public struct JacobianFactor: LinearFactor {

  @differentiable(wrt: values)
  public func error(_ indices: [Int], values: Tensor<ScalarType>) -> ScalarType {
    ScalarType.zero
  }
  
  public var keys: Array<Int>
  
  typealias Output = Tensor<ScalarType>
//  
//  /// Comparable to the `*` operator in GTSAM
//  @differentiable
//  static func * (lhs: JacobianFactor, rhs: Self.Input) -> Self.Output
}
