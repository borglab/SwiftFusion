// Copyright 2020 Penguin Authors
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

public struct TypewiseBufferKey<Element> {
  var elementType: Type<Element> { .init() }
  var bufferID: TypeID { .init(Element.self) }

  func map<NewElement>(_: Type<NewElement> = .init()) -> TypeMappedBufferKey<NewElement> {
    .init(bufferID: bufferID)
  }
}

public struct TypeMappedBufferKey<Element> {
  let bufferID: TypeID
  
  var elementType: Type<Element> { .init() }
  var unmapped: TypewiseBufferKey<Element> { .init() }
  
  init(bufferID: TypeID) {
    self.bufferID = bufferID
  }
  
  init(_ source: TypewiseBufferKey<Element>) {
    self = source.map(Type<Element>())
  }
}

public struct TypewiseElementID<Element> {
  let position: ArrayBuffer<Element>.Index
  
  var elementType: Type<Element> { .init() }
  var bufferID: TypeID { .init(Element.self) }
  var bufferKey: TypewiseBufferKey<Element> { .init() }

  func map<NewElement>(_: Type<NewElement> = .init()) -> TypeMappedElementID<NewElement> {
    .init(bufferID: bufferID, position: position)
  }
}

public struct TypeMappedElementID<Element> {
  let bufferID: TypeID
  let position: ArrayBuffer<Element>.Index
  
  var elementType: Type<Element> { .init() }
  var bufferKey: TypeMappedBufferKey<Element> { .init(bufferID: bufferID) }
  var unmapped: TypewiseElementID<Element> { .init(position: position) }
  
  init(bufferID: TypeID, position: ArrayBuffer<Element>.Index) {
    self.bufferID = bufferID
    self.position = position
  }
  
  init(_ source: TypewiseElementID<Element>) {
    self = source.map(Type<Element>())
  }
}

extension Dictionary {
  subscript(existingKey k: Key) -> Value {
    get { self[k]! }
    _modify {
      yield &values[index(forKey: k)!]
    }
  }
}

public struct TypewiseStorage<Dispatch: AnyObject> {
  public typealias ElementID<T> = TypewiseElementID<T>
  public typealias BufferKey<T> = TypewiseBufferKey<T>

  private var base: TypeMappedStorage<Dispatch>
  
  public mutating func store<T>(
    _ x: T,
    makeBuffer: (ArrayBuffer<T>) -> AnyArrayBuffer<Dispatch>
  ) -> ElementID<T>
  {
    base.store(x, inBuffer: TypeID(T.self), makeBuffer: makeBuffer).unmapped
  }
  
  public subscript<Element>(id: ElementID<Element>) -> Element {
    get {
      base[id.map()]
    }
    _modify {
      yield &base[id.map()]
    }
  }

  public subscript<Element>(k: BufferKey<Element>) -> ArrayBuffer<Element> {
    get {
      base[k.map()]
    }
    _modify {
      yield &base[k.map()]
    }
  }
}

public struct TypeMappedStorage<Dispatch: AnyObject> {
  public typealias ElementID<T> = TypeMappedElementID<T>
  public typealias BufferKey<T> = TypeMappedBufferKey<T>
  
  private var buffers: [TypeID: AnyArrayBuffer<Dispatch>] = [:]
  
  public mutating func store<T>(
    _ x: T, inBuffer bufferID: TypeID,
    makeBuffer: (ArrayBuffer<T>) -> AnyArrayBuffer<Dispatch>
  ) -> ElementID<T>
  {
    if let position
      = buffers[bufferID, default: makeBuffer(.init())][elementType: Type<T>()]?.append(x)
    {
      return .init(bufferID: bufferID, position: position)
    }
    fatalError(
      """
      Can't insert \(T.self) in buffer of \(buffers[bufferID]!.elementType) \
      with id \(bufferID).
      """)
  }
  
  public subscript<Element>(id: ElementID<Element>) -> Element {
    get {
      self[id.bufferKey][id.position]
    }
    _modify {
      yield &self[id.bufferKey][id.position]
    }
  }

  public subscript<Element>(k: BufferKey<Element>) -> ArrayBuffer<Element> {
    get {
      buffers[existingKey: k.bufferID][existingElementType: k.elementType]
    }
    _modify {
      yield &buffers[existingKey: k.bufferID][existingElementType: k.elementType]
    }
  }
}
