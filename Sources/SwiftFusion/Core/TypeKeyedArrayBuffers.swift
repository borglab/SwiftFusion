// Copyright 2020 SwiftFusion Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import PenguinStructures

/// Types that store a heterogeneous collection of `ArrayBuffer`s (in the form of
/// `AnyArrayBuffer<Dispatch>`) with common capabilities, keyed by `TypeID`.
public protocol TypeKeyedArrayBuffersProtocol {
  /// A dispatch table for the common capabilites of each buffer type.
  associatedtype Dispatch: AnyObject

  /// A key that can be used to look up an untyped `ArrayBuffer` (i.e. `AnyArrayBuffer<Dispatch>`)
  typealias UntypedKey = TypeID
  
  /// (Internal only) The backing store.
  var _storage: [UntypedKey: AnyArrayBuffer<Dispatch>] { get set }
  
  /// (Internal only) Creates an instance with the given backing store.
  init(_storage: Storage)
}

/// A mapping from instances of `TypeID` to instances of `AnyArrayBuffer<Dispatch>`
public struct TypeKeyedArrayBuffers<Dispatch: AnyObject>: TypeKeyedArrayBuffersProtocol {
  public typealias Dispatch = Dispatch
  public var _storage: Storage
  public init(_storage: Storage) { self._storage = _storage }
}

extension TypeKeyedArrayBuffersProtocol {
  /// Creates an empty instance
  public init() { self.init(_storage: [:]) }

  /// Representation of the stored `AnyArrayBuffer`s.
  public typealias AnyBuffers = Storage.Values
  
  /// Representation of the `TypeID`s that index the stored `AnyArrayBuffer`s.
  public typealias UntypedKeys = Storage.Keys
  
  /// The stored `AnyArrayBuffer`s.
  public var anyBuffers: AnyBuffers {
    get { _storage.values }
    _modify { yield &_storage.values}
  }

  /// the `TypeID`s that index the stored `AnyArrayBuffer`s.
  private var untypedKeys: UntypedKeys { _storage.keys }

  /// Returns `true` iff `self` and other have the same key values, and for each key, the same
  /// number of elements in the corresponding array.
  internal func hasSameStructure<D>(as other: TypeKeyedArrayBuffers<D>) -> Bool {
    if _storage.count != other._storage.count { return false }
    return _storage.allSatisfy { k, v in  other._storage[k]?.count == v.count }
  }

  /// Returns a mapping from each key `k` of `self` into `bufferTransform(self[k])`.
  public func mapBuffers<NewDispatch>(
    _ bufferTransform: (AnyArrayBuffer<Dispatch>) throws -> AnyArrayBuffer<NewDispatch>
  ) rethrows -> TypeKeyedArrayBuffers<NewDispatch> {
    try .init(_storage: _storage.mapValues(bufferTransform))
  }

  /// Returns a mapping from each key `k` of `self` into the corresponding array, transformed by
  /// `bufferTransform`.
  public func compactMapBuffers<NewDispatch>(
    _ bufferTransform: (AnyArrayBuffer<Dispatch>) throws -> AnyArrayBuffer<NewDispatch>?
  ) rethrows -> TypeKeyedArrayBuffers<NewDispatch> {
    try .init(_storage: _storage.compactMapValues(bufferTransform))
  }

  /// Invokes `update` on each buffer of self, passing the buffer having the same `key` in
  /// `parameter` as a second argument.
  ///
  /// - Requires: `self.hasSameStructure(as: parameter)`
  public mutating func updateBuffers<OtherDispatch>(
    homomorphicArgument parameter: TypeKeyedArrayBuffers<OtherDispatch>,
    _ update: (
      _ myBuffer: inout AnyArrayBuffer<Dispatch>,
      _ parameter: AnyArrayBuffer<OtherDispatch>) throws -> Void
  ) rethrows {
    precondition(
      _storage.count == parameter._storage.count,
      "parameter must have same structure as `self`")
    try _storage.updateValues { k, target, _ in
      guard let p = parameter._storage[k] else {
        fatalError("parameter must have same structure as `self`")
      }
      try update(&target, p)
    }
  }
}
