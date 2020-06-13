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

/// A heterogeneous collection of vectors.
///
/// TODO: This really should be a different type from `VariableAssignments` because (1) we know
/// statically that `AllVectors` only contains vector values, and (2) we use `AllVectors` in some
/// places for things other than variable assignments (e.g. a collection of error vectors is one
/// error vector per factor).
public typealias AllVectors = VariableAssignments

/// Vector operations.
// TODO: There are some mutating operations here that copy, mutate, and write back. Make these
// more efficient.
extension AllVectors {
  /// Returns the squared norm of `self`, where `self` is viewed as a vector in the vector space
  /// direct sum of all the variables.
  ///
  /// Precondition: All the variables in `self` are vectors.
  public var squaredNorm: Double {
    return storage.values.lazy
      .map { $0.storage as! AnyVectorStorage }
      .map { $0.dot($0) }
      .reduce(0, +)
  }

  /// Returns the scalar product of `lhs` with `rhs`, where `rhs` is viewed as a vector in the
  /// vector space direct sum of all the variables.
  ///
  /// Precondition: All the variables in `rhs` are vectors.
  public static func * (_ lhs: Double, _ rhs: Self) -> Self {
    VariableAssignments(storage: rhs.storage.mapValues { value in
      var vector = value.cast(to: AnyVectorStorage.self)!
      vector.scale(by: lhs)
      return AnyArrayBuffer(vector)
    })
  }

  /// Returns the vector sum of `lhs` with `rhs`, where `lhs` and `rhs` are viewed as vectors in the
  /// vector space direct sum of all the variables.
  ///
  /// Precondition: All the elements in `lhs` and `rhs` are vectors. `lhs` and `rhs` have
  /// assignments for exactly the same sets of variables.
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    let r = Dictionary(uniqueKeysWithValues: lhs.storage.map {
      (key, value) -> (ObjectIdentifier, AnyArrayBuffer<AnyArrayStorage>) in
      var resultVector = value.cast(to: AnyVectorStorage.self)!
      let rhsVector = rhs.storage[key]!.cast(to: AnyVectorStorage.self)!
      resultVector.add(rhsVector)
      return (key, AnyArrayBuffer(resultVector))
    })
    return VariableAssignments(storage: r)
  }

  /// Returns the `ErrorVector` from the `perFactorID`-th factor of type `F`.
  subscript<F: NewLinearizableFactor>(_ perFactorID: Int, factorType _: F.Type) -> F.ErrorVector {
    return storage[ObjectIdentifier(F.self)]!
      .withUnsafeBufferPointer(assumingElementType: F.ErrorVector.self) { b in
        return b[perFactorID]
      }
  }
}
