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

import _Differentiation
import PenguinStructures

extension TypeKeyedArrayBuffers: Equatable where ElementAPI: VectorArrayDispatch {
  public static func == (lhs: Self, rhs: Self) -> Bool {
    lhs.firstBufferKey(homomorphicArgument: rhs) { $0 != $1 } == nil
  }
}
  
extension TypeKeyedArrayBuffers: AdditiveArithmetic where ElementAPI == VectorArrayDispatch {
  /// Returns the vector sum of `lhs` with `rhs`, where `lhs` and `rhs` are viewed as vectors in the
  /// vector space direct sum of all the variables.
  ///
  /// Precondition: `lhs` and `rhs` have assignments for exactly the same sets of variables.
  public static func + (lhs: Self, rhs: Self) -> Self {
    lhs.updatedBuffers(homomorphicArgument: rhs) { $0 + $1 }
  }

  public static func += (lhs: inout Self, rhs: Self) {
    lhs.updateBuffers(homomorphicArgument: rhs) { $0 += $1 }
  }

  public static func - (lhs: Self, rhs: Self) -> Self {
    lhs.updatedBuffers(homomorphicArgument: rhs) { $0 - $1 }
  }

  public static func -= (lhs: inout Self, rhs: Self) {
    lhs.updateBuffers(homomorphicArgument: rhs) { $0 -= $1 }
  }

  public static var zero: Self { .init() }
}

extension TypeKeyedArrayBuffers: Differentiable where ElementAPI: DifferentiableArrayDispatch {
  public typealias TangentVector = MappedArrayBuffers<VectorArrayDispatch>
  public mutating func move(along offset: TangentVector) {
    updateBuffers(homomorphicArgument: offset) { $0.move(along: $1) }
  }

  public var zeroTangentVectorInitializer: () -> TangentVector {
    { .init() }
  }
}
