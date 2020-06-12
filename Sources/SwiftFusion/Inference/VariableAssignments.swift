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

/// An identifier of a given abstract value with the value's type attached
///
/// - Parameter Value: the type of value this ID refers to.
/// - Parameter PerTypeID: a type that, given `Value`, identifies a given
///   logical value of that type.
///
/// Note: This is just a temporary placeholder until we get the real `TypedID` in penguin.
public struct TypedID<Value, PerTypeID: Equatable> {
  /// A specifier of which logical value of type `value` is being identified.
  public let perTypeID: PerTypeID

  /// Creates an instance indicating the given logical value of type `Value`.
  public init(_ perTypeID: PerTypeID) { self.perTypeID = perTypeID }
}

/// Assignments of values to factor graph variables.
public struct VariableAssignments {
  /// Dictionary from variable type to contiguous storage for that type.
  var contiguousStorage: [ObjectIdentifier: AnyArrayBuffer<AnyArrayStorage>] = [:]

  /// Creates an empty instance.
  public init() {}

  internal init(contiguousStorage: [ObjectIdentifier: AnyArrayBuffer<AnyArrayStorage>]) {
    self.contiguousStorage = contiguousStorage
  }

  /// Stores `value` as the assignment of a new variable, and returns the new variable's id.
  public mutating func store<T>(_ value: T) -> TypedID<T, Int> {
    let perTypeID = contiguousStorage[
      ObjectIdentifier(T.self),
      default: AnyArrayBuffer(ArrayBuffer<ArrayStorage<T>>())
    ].append(value)
    assert(type(of: contiguousStorage[ObjectIdentifier(T.self)]!.storage) == ArrayStorage<T>.self)
    return TypedID(perTypeID)
  }

  /// Stores `value` as the assignment of a new variable, and returns the new variable's id.
  public mutating func store<T: Differentiable>(_ value: T) -> TypedID<T, Int>
    where T.TangentVector: EuclideanVector
  {
    let perTypeID = contiguousStorage[
      ObjectIdentifier(T.self),
      default: AnyArrayBuffer(ArrayBuffer<DifferentiableArrayStorage<T>>())
    ].append(value)
    assert(
      type(of: contiguousStorage[ObjectIdentifier(T.self)]!.storage)
        == DifferentiableArrayStorage<T>.self
    )
    return TypedID(perTypeID)
  }

  /// Stores `value` as the assignment of a new variable, and returns the new variable's id.
  public mutating func store<T: EuclideanVector>(_ value: T) -> TypedID<T, Int> {
    let perTypeID = contiguousStorage[
      ObjectIdentifier(T.self),
      default: AnyArrayBuffer(ArrayBuffer<VectorArrayStorage<T>>())
    ].append(value)
    assert(
      type(of: contiguousStorage[ObjectIdentifier(T.self)]!.storage)
        == VectorArrayStorage<T>.self
    )
    return TypedID(perTypeID)
  }

  /// Traps with an error indicating that an attempt was made to access a stored
  /// variable of a type that is not represented in `self`.
  private static var noSuchType: AnyArrayBuffer<AnyArrayStorage> {
    fatalError("No such stored variable type")
  }

  /// Accesses the stored value with the given ID.
  public subscript<T>(id: TypedID<T, Int>) -> T {
    _read {
      yield contiguousStorage[ObjectIdentifier(T.self), default: Self.noSuchType]
        .withUnsafeRawPointerToElements { p in
          p.assumingMemoryBound(to: T.self).advanced(by: id.perTypeID).pointee
        }
    }
    _modify {
      defer { _fixLifetime(self) }
      yield &contiguousStorage[ObjectIdentifier(T.self), default: Self.noSuchType]
        .withUnsafeMutableRawPointerToElements { p in
          p.assumingMemoryBound(to: T.self).advanced(by: id.perTypeID)
        }
        .pointee
    }
  }
}

/// Differentiable operations.
extension VariableAssignments {
  /// Assignment of zero to all the variables in `GenericFactorGraph.linearized(at: self)`.
  public var linearizedZero: VariableAssignments {
    let r = Dictionary(uniqueKeysWithValues: contiguousStorage.compactMap {
      (key, value) -> (ObjectIdentifier, AnyArrayBuffer<AnyArrayStorage>)? in
      guard let differentiableValue = value.cast(to: AnyDifferentiableStorage.self) else {
        return nil
      }
      return (
        differentiableValue.tangentIdentifier,
        AnyArrayBuffer(differentiableValue.zeroTangent)
      )
    })
    return VariableAssignments(contiguousStorage: r)
  }

  /// Moves each differentiable variable along the corresponding element of `direction`.
  ///
  /// See `GenericFactorGraph.linearized(at:)` for documentation about the correspondence between
  /// differentiable variables and their linearizations.
  public mutating func move(along direction: VariableAssignments) {
    contiguousStorage = contiguousStorage.mapValues { value in
      guard var diffVal = value.cast(to: AnyDifferentiableStorage.self) else {
        return value
      }
      guard let dirElem = direction.contiguousStorage[diffVal.tangentIdentifier] else {
        return value
      }
      diffVal.move(along: dirElem)
      return AnyArrayBuffer(diffVal)
    }
  }

  /// See `move(along:)`.
  public func moved(along direction: VariableAssignments) -> Self {
    var result = self
    result.move(along: direction)
    return result
  }
}

/// Vector operations.
// TODO: These currently have a precondition that all stored values are vectors. We could improve
// this by creating and using a "Vectors" type that is statically known to contain only vectors.
extension VariableAssignments {
  /// Returns the squared norm of `self`, where `self` is viewed as a vector in the vector space
  /// direct sum of all the variables.
  ///
  /// Precondition: All the variables in `self` are vectors.
  public var squaredNorm: Double {
    return contiguousStorage.values.lazy
      .map { $0.storage as! AnyVectorStorage }
      .map { $0.dot($0) }
      .reduce(0, +)
  }

  /// Returns the scalar product of `lhs` with `rhs`, where `rhs` is viewed as a vector in the
  /// vector space direct sum of all the variables.
  ///
  /// Precondition: All the variables in `rhs` are vectors.
  public static func * (_ lhs: Double, _ rhs: Self) -> Self {
    VariableAssignments(contiguousStorage: rhs.contiguousStorage.mapValues { value in
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
    let r = Dictionary(uniqueKeysWithValues: lhs.contiguousStorage.map {
      (key, value) -> (ObjectIdentifier, AnyArrayBuffer<AnyArrayStorage>) in
      var resultVector = value.cast(to: AnyVectorStorage.self)!
      let rhsVector = rhs.contiguousStorage[key]!.cast(to: AnyVectorStorage.self)!
      resultVector.add(rhsVector)
      return (key, AnyArrayBuffer(resultVector))
    })
    return VariableAssignments(contiguousStorage: r)
  }
}
