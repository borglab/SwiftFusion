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
  private var _storage: Storage
  private init(_storage: Storage) { self._storage = _storage }
  
  /// A key that can be used to look up an untyped `ArrayBuffer` (i.e. `AnyArrayBuffer<Dispatch>`)
  fileprivate typealias UntypedKey = TypeID

  /// The type of the backing store
  private typealias Storage = [UntypedKey: AnyArrayBuffer<Dispatch>]
  
  /// A dispatch table for the common capabilites of each buffer type.
  public typealias Dispatch = Dispatch
}

extension TypeKeyedArrayBuffers {
  /// A key for accessing a particular `ArrayBuffer`.
  ///
  /// The `Element` type of the `ArrayBuffer` is communicated by the `Element` type parameter.
  public struct BufferKey<Element> {
    /// The untyped key associated with the buffer looked up with this key.
    fileprivate let untyped: UntypedKey
    
    /// The `Element` type of buffers looked up with this key.
    fileprivate var elementType: Type<Element> { .init() }
  }

  /// A key for accessing a `store`d element.
  ///
  /// Note that only elements directly inserted via a call to `store` (as opposed to those generated
  /// via `mapBuffers`) can be accessed this way.
  public struct KeyToStored<Element> {
    /// The index where the `Element` is stored in its `ArrayBuffer`.
    fileprivate let indexInBuffer: Int
      
    /// The key to the buffer in which the accessed element is stored.
    fileprivate var buffer: BufferKey<Element> { .init(untyped: Type<Element>.id) }
    
    /// The `Element` type of buffers looked up with this key.
    fileprivate var elementType: Type<Element> { .init() }
  }

  /// Creates an empty instance
  public init() { self.init(_storage: [:]) }

  /// Representation of the stored `AnyArrayBuffer`s.
  public typealias AnyBuffers = Dictionary<TypeID, AnyArrayBuffer<Dispatch>>.Values
  
  /// Representation of the `TypeID`s that index the stored `AnyArrayBuffer`s.
  private typealias UntypedKeys = Storage.Keys
  
  /// The stored `AnyArrayBuffer`s.
  public var anyBuffers: AnyBuffers {
    get { _storage.values }
    _modify { yield &_storage.values}
  }

  /// The keys that index the stored `AnyArrayBuffer`s.
  private var untypedKeys: UntypedKeys { _storage.keys }

  /// Accesses the buffer associated with `k`.
  ///
  /// - Requires: the buffer associated with `k.untyped` stores elements of type `Element`.
  public subscript<Element>(k: BufferKey<Element>) -> ArrayBuffer<Element> {
    get {
      _storage[existingKey: k.untyped][existingElementType: k.elementType]
    }
    _modify {
      yield &_storage[existingKey: k.untyped][existingElementType: k.elementType]
    }
  }

  /// Returns `true` iff `self` and other have the same key values, and for each key, the same
  /// number of elements in the corresponding array.
  public func hasSameStructure<D>(as other: TypeKeyedArrayBuffers<D>) -> Bool {
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

  /// Stores `newElement`, returning its key.
  ///
  /// - Parameter upcast: used to erase the type of a newly-created buffer if none exists with the
  ///   right element type.
  public mutating func store<Element>(
    _ newElement: Element, upcast: (ArrayBuffer<Element>) -> AnyArrayBuffer<Dispatch>
  ) -> KeyToStored<Element>
  {
    let bufferKey = Type<Element>.id
    if let i = _storage[bufferKey, default: upcast(.init())][elementType: Type<Element>()]?
         .append(newElement)
    {
      return .init(indexInBuffer: i)
    }
    fatalError(
      """
      Can't insert \(Element.self) in buffer of \(_storage[bufferKey]!.elementType) \
      with key \(Element.self).
      """)
  }

  /// Accesses the element for which invoking `store` returned `id`.
  public subscript<Element>(id: KeyToStored<Element>) -> Element {
    get {
      self[id.buffer][id.indexInBuffer]
    }
    _modify {
      yield &self[id.buffer][id.indexInBuffer]
    }
  }
}
