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
  var storage: [ObjectIdentifier: AnyArrayBuffer<AnyArrayStorage>] = [:]

  /// Creates an empty instance.
  public init() {}

  /// Creates an instance backed by the given `storage`.
  internal init(storage: [ObjectIdentifier: AnyArrayBuffer<AnyArrayStorage>]) {
    self.storage = storage
  }

  /// Stores `value` as the assignment of a new variable, and returns the new variable's id.
  public mutating func store<T>(_ value: T) -> TypedID<T, Int> {
    let perTypeID = storage[
      ObjectIdentifier(T.self),
      default: AnyArrayBuffer(ArrayBuffer<ArrayStorage<T>>())
    ].append(value)
    assert(type(of: storage[ObjectIdentifier(T.self)]!.storage) == ArrayStorage<T>.self)
    return TypedID(perTypeID)
  }

  /// Stores `value` as the assignment of a new variable, and returns the new variable's id.
  public mutating func store<T: Differentiable>(_ value: T) -> TypedID<T, Int>
    where T.TangentVector: EuclideanVectorN
  {
    let perTypeID = storage[
      ObjectIdentifier(T.self),
      default: AnyArrayBuffer(ArrayBuffer<DifferentiableArrayStorage<T>>())
    ].append(value)
    assert(
      type(of: storage[ObjectIdentifier(T.self)]!.storage)
        == DifferentiableArrayStorage<T>.self
    )
    return TypedID(perTypeID)
  }

  /// Stores `value` as the assignment of a new variable, and returns the new variable's id.
  public mutating func store<T: EuclideanVectorN>(_ value: T) -> TypedID<T, Int> {
    let perTypeID = storage[
      ObjectIdentifier(T.self),
      default: AnyArrayBuffer(ArrayBuffer<VectorArrayStorage<T>>())
    ].append(value)
    assert(
      type(of: storage[ObjectIdentifier(T.self)]!.storage)
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
      yield storage[ObjectIdentifier(T.self), default: Self.noSuchType]
        .withUnsafeRawPointerToElements { p in
          p.assumingMemoryBound(to: T.self).advanced(by: id.perTypeID).pointee
        }
    }
    _modify {
      defer { _fixLifetime(self) }
      yield &storage[ObjectIdentifier(T.self), default: Self.noSuchType]
        .withUnsafeMutableRawPointerToElements { p in
          p.assumingMemoryBound(to: T.self).advanced(by: id.perTypeID)
        }
        .pointee
    }
  }
}

/// Differentiable operations.
// TODO: There are some mutating operations here that copy, mutate, and write back. Make these
// more efficient.
extension VariableAssignments {
  /// For each differentiable value in `self`, the zero value of its tangent vector.
  public var tangentVectorZeros: AllVectors {
    let r = Dictionary(uniqueKeysWithValues: storage.compactMap {
      (key, value) -> (ObjectIdentifier, AnyArrayBuffer<AnyArrayStorage>)? in
      guard let differentiableValue = value.cast(to: AnyDifferentiableStorage.self) else {
        return nil
      }
      return (
        differentiableValue.tangentIdentifier,
        AnyArrayBuffer(differentiableValue.zeroTangent)
      )
    })
    return AllVectors(storage: r)
  }

  /// Moves each differentiable variable along the corresponding element of `direction`.
  ///
  /// See `NewFactorGraph.linearized(at:)` for documentation about the correspondence between
  /// differentiable variables and their linearizations.
  public mutating func move(along direction: AllVectors) {
    storage = storage.mapValues { value in
      guard var diffVal = value.cast(to: AnyDifferentiableStorage.self) else {
        return value
      }
      guard let dirElem = direction.storage[diffVal.tangentIdentifier] else {
        return value
      }
      diffVal.move(along: dirElem)
      return AnyArrayBuffer(diffVal)
    }
  }

  /// See `move(along:)`.
  public func moved(along direction: AllVectors) -> Self {
    // TODO: Make sure that this is efficient when we have a unique reference.
    var result = self
    result.move(along: direction)
    return result
  }
}
