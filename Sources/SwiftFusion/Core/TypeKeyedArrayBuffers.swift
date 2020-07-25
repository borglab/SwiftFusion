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

/// A mapping from instances of `TypeID` to instances of `AnyArrayBuffer<Dispatch>`
public struct TypeKeyedArrayBuffers<Dispatch: AnyObject> {
  internal typealias Storage = [TypeID: AnyArrayBuffer<Dispatch>]
  private var storage: Storage
  
  internal init(_ storage: Storage) { self.storage = storage }
  public init() { storage = [:] }
}

extension TypeKeyedArrayBuffers {
  public typealias Buffers = Dictionary<TypeID, AnyArrayBuffer<Dispatch>>.Values
  public typealias Keys = Dictionary<TypeID, AnyArrayBuffer<Dispatch>>.Keys
  public typealias Key = TypeID
  
  public var buffers: Buffers {
    get { storage.values }
    _modify { yield &storage.values}
  }
  private var keys: Keys { storage.keys }

  /// Returns `true` iff `self` and other have the same key values, and for each key, the same
  /// number of elements in the corresponding array.
  internal func hasSameStructure<D>(as other: TypeKeyedArrayBuffers<D>) -> Bool {
    if storage.count != other.storage.count { return false }
    return storage.allSatisfy { k, v in  other.storage[k]?.count == v.count }
  }

  /// Returns a mapping from each key `k` of `self` into `bufferTransform(self[k])`.
  public func mapBuffers<NewDispatch>(
    _ bufferTransform: (AnyArrayBuffer<Dispatch>) throws -> AnyArrayBuffer<NewDispatch>
  ) rethrows -> TypeKeyedArrayBuffers<NewDispatch> {
    try .init(storage.mapValues(bufferTransform))
  }

  /// Returns a mapping from each key `k` of `self` into the corresponding array, transformed by
  /// `bufferTransform`.
  public func compactMapBuffers<NewDispatch>(
    _ bufferTransform: (AnyArrayBuffer<Dispatch>) throws -> AnyArrayBuffer<NewDispatch>?
  ) rethrows -> TypeKeyedArrayBuffers<NewDispatch> {
    try .init(storage.compactMapValues(bufferTransform))
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
      storage.count == parameter.storage.count,
      "parameter must have same structure as `self`")
    try storage.updateValues { k, target, _ in
      guard let p = parameter.storage[k] else {
        fatalError("parameter must have same structure as `self`")
      }
      try update(&target, p)
    }
  }
}
