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
public struct TypedID<Value, PerTypeID: Equatable>: Equatable {
  /// A specifier of which logical value of type `value` is being identified.
  public let perTypeID: PerTypeID

  /// Creates an instance indicating the given logical value of type `Value`.
  public init(_ perTypeID: PerTypeID) { self.perTypeID = perTypeID }
}

/// Assignments of values to factor graph variables.
public struct VariableAssignments {
  /// Dictionary from variable type to contiguous storage for that type.
  var storage: [ObjectIdentifier: AnyElementArrayBuffer] = [:]

  /// Creates an empty instance.
  public init() {}

  /// Creates an instance backed by the given `storage`.
  internal init(storage: [ObjectIdentifier: AnyElementArrayBuffer]) {
    self.storage = storage
  }

  /// Stores `value` as the assignment of a new variable, and returns the new variable's id.
  public mutating func store<T>(_ value: T) -> TypedID<T, Int> {
    let perTypeID = storage[
      ObjectIdentifier(T.self),
      default: AnyElementArrayBuffer(ArrayBuffer<T>())
    ].unsafelyAppend(value)
    return TypedID(perTypeID)
  }

  /// Stores `value` as the assignment of a new variable, and returns the new variable's id.
  public mutating func store<T: Differentiable>(_ value: T) -> TypedID<T, Int>
    where T.TangentVector: EuclideanVectorN
  {
    let perTypeID = storage[
      ObjectIdentifier(T.self),
      // Note: This is a safe upcast.
      default: AnyElementArrayBuffer(
        unsafelyCasting: AnyDifferentiableArrayBuffer(ArrayBuffer<T>()))
    ].unsafelyAppend(value)
    return TypedID(perTypeID)
  }

  /// Stores `value` as the assignment of a new variable, and returns the new variable's id.
  public mutating func store<T: EuclideanVectorN>(_ value: T) -> TypedID<T, Int> {
    let perTypeID = storage[
      ObjectIdentifier(T.self),
      // Note: This is a safe upcast.
      default: AnyElementArrayBuffer(unsafelyCasting: AnyVectorArrayBuffer(ArrayBuffer<T>()))
    ].unsafelyAppend(value)
    return TypedID(perTypeID)
  }

  /// Traps with an error indicating that an attempt was made to access a stored
  /// variable of a type that is not represented in `self`.
  private static var noSuchType: AnyElementArrayBuffer {
    fatalError("No such stored variable type")
  }

  /// Accesses the stored value with the given ID.
  public subscript<T>(id: TypedID<T, Int>) -> T {
    _read {
      let array = storage[ObjectIdentifier(T.self), default: Self.noSuchType]
      yield ArrayBuffer<T>(unsafelyDowncasting: array)
        .withUnsafeBufferPointer { b in b[id.perTypeID] }
    }
    _modify {
      defer { _fixLifetime(self) }
      let array = storage[ObjectIdentifier(T.self), default: Self.noSuchType]
      yield &(array.storage as! ArrayStorage<T>)
        .withUnsafeMutableBufferPointer { b in b.baseAddress.unsafelyUnwrapped + id.perTypeID }
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
      (key, value) -> (ObjectIdentifier, AnyElementArrayBuffer)? in
      guard let differentiableValue = AnyArrayBuffer<DifferentiableArrayDispatch>(value) else {
        return nil
      }
      return (
        ObjectIdentifier(differentiableValue.tangentVectorType),
        // Note: This is a safe upcast.
        AnyElementArrayBuffer(unsafelyCasting: differentiableValue.tangentVectorZeros)
      )
    })
    return AllVectors(storage: r)
  }

  /// Moves each differentiable variable along the corresponding element of `direction`.
  ///
  /// See `FactorGraph.linearized(at:)` for documentation about the correspondence between
  /// differentiable variables and their linearizations.
  public mutating func move(along direction: AllVectors) {
    storage = storage.mapValues { value in
      guard var diffVal = AnyArrayBuffer<DifferentiableArrayDispatch>(value) else {
        return value
      }
      guard let dirElem = direction.storage[ObjectIdentifier(diffVal.tangentVectorType)] else {
        return value
      }
      diffVal.move(along: dirElem)
      // Note: This is a safe upcast.
      return AnyElementArrayBuffer(unsafelyCasting: diffVal)
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
