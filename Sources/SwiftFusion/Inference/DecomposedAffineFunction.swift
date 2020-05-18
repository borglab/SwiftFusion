// Copyright 2020 The SwiftFusion Authors. All Rights Reserved.
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

/// An affine function decomposed into its linear and bias components.
public protocol DecomposedAffineFunction {
  associatedtype Input: EuclideanVectorSpace
  associatedtype Output: EuclideanVectorSpace

  /// Apply the function to `x`.
  ///
  /// This is equal to `applyLinearForward(x) + bias`.
  ///
  /// Note: A default implementation is provided, but conforming types may provide a more efficient
  /// implementation.
  func callAsFunction(_ x: Input) -> Output

  /// The linear component of the affine function.
  func applyLinearForward(_ x: Input) -> Output

  /// The linear adjoint of the linear component of the affine function.
  func applyLinearAdjoint(_ y: Output) -> Input

  /// The bias component of the affine function.
  ///
  /// This is equal to `applyLinearForward(Input.zero)`.
  var bias: Output { get }
}

extension DecomposedAffineFunction {
  public func callAsFunction(_ x: Input) -> Output {
    return applyLinearForward(x) + bias
  }
}
