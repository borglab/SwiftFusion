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
public typealias Error = Vector

/// Collection of all errors returned by a Factor Graph
public struct Errors {
  public var values: Array<Error>.DifferentiableView

  /// Creates empty `Errors`.
  public init() {
    self.values = Array.DifferentiableView()
  }

  /// Creates `Errors` containing the given `errors`.
  public init(_ errors: [Error]) {
    self.values = Array.DifferentiableView(errors)
  }

  public static func += (_ lhs: inout Self, _ rhs: [Error]) {
    lhs.values.base += rhs
  }
}

/// Extending Array for Error type
/// This simplifies the implementation for `Errors`, albeit in a less self-contained manner
/// TODO: change this to a concrete `struct Errors` and implement all the protocols
extension Errors: EuclideanVectorSpace {

  // Note: Requirements of `Differentiable`, `AdditiveArithmetic`, and `VectorProtocol` are automatically
  // synthesized. Yay!

  /// Calculates the L2 norm
  public var squaredNorm: Double {
    get {
      values.base.map { $0.squared().sum() }.reduce(0.0, { $0 + $1 })
    }
  }
}
