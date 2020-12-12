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

import PenguinStructures

extension TypeKeyedArrayBuffers where ElementAPI == VectorArrayDispatch {
  /// Returns the squared norm of `self`, where `self` is viewed as a vector in the vector space
  /// direct sum of all the variables.
  ///
  /// Precondition: All the variables in `self` are vectors.
  public var squaredNorm: Double {
    anyBuffers.lazy.map { $0.dot($0) }.reduce(0, +)
  }

  /// Returns the scalar product of `lhs` with `rhs`, where `rhs` is viewed as a vector in the
  /// vector space direct sum of all the variables.
  ///
  /// Precondition: All the variables in `rhs` are vectors.
  public static func * (_ lhs: Double, _ rhs: Self) -> Self {
    let r = rhs.mapBuffers { lhs * $0 }
    return r
  }

  /// Returns the vector sum of `lhs` with `rhs`, where `lhs` and `rhs` are viewed as vectors in the
  /// vector space direct sum of all the variables.
  ///
  /// Precondition: All the elements in `lhs` and `rhs` are vectors. `lhs` and `rhs` have
  /// assignments for exactly the same sets of variables.
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    lhs.updatedBuffers(homomorphicArgument: rhs, +)
  }

  /*
  /// Stores an `ErrorVector` for a factor of type `F`.
  public mutating func store<F: LinearizableFactor>(_ value: F.ErrorVector, factorType _: F.Type) {
    _ = _storage[
      Type<F>.id,
      // Note: This is a safe upcast.
      default: AnyVectorArrayBuffer(ArrayBuffer<F.ErrorVector>())
    ].unsafelyAppend(value)
  }
  
  /// Returns the `ErrorVector` from the `perFactorID`-th factor of type `F`.
  public subscript<F: VectorFactor>(_ perFactorID: Int, factorType _: F.Type)
    -> F.ErrorVector
  {
    let array = _storage[Type<F>.id].unsafelyUnwrapped
    return ArrayBuffer<F.ErrorVector>(unsafelyDowncasting: array).withUnsafeBufferPointer { b in
      b[perFactorID]
    }
  }
   */
}
