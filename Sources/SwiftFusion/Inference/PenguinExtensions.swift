// TODO: Move all these to Penguin.

import PenguinStructures

extension AnyArrayBuffer {
  /// Appends `x`, returning the index of the appended element.
  ///
  /// - Requires: `self.elementType == T.self`.
  mutating func unsafelyAppend<T>(_ x: T) -> Int {
    self[unsafelyAssumingElementType: Type<T>()].append(x)
  }
}

extension UnsafeRawPointer {
  /// Returns the `T` to which `self` points.
  internal subscript<T>(as _: Type<T>) -> T {
    self.assumingMemoryBound(to: T.self).pointee
  }
}

extension UnsafeMutableRawPointer {
  /// Returns the `T` to which `self` points.
  internal subscript<T>(as _: Type<T>) -> T {
    get {
      self.assumingMemoryBound(to: T.self).pointee
    }
    _modify {
      yield &self.assumingMemoryBound(to: T.self).pointee
    }
  }
}

extension AnyArrayBuffer {
  /// Accesses `self` as an `ArrayBuffer<Element>` if `Element.self == elementType`.
  ///
  /// - Writing `nil` removes all elements of `self`.
  /// - Where `Element.self != elementType`:
  ///   - reading returns `nil`.
  ///   - writing changes the type of elements stored in `self`.
  ///
  public subscript<Element>(elementType _: Type<Element>) -> ArrayBuffer<Element>? {
    get {
      ArrayBuffer<Element>(self)
    }
    
    _modify {
      // TODO: check for spurious ARC traffic
      var me = ArrayBuffer<Element>(self)
      if me != nil {
        // Don't retain an additional reference during this mutation, to prevent needless CoW.
        self.storage = nil
      }
      yield &me
      if let written = me {
        self.storage = .init(written.storage)
      }
      else {
        self.storage = .init(ArrayStorage<Element>(minimumCapacity: 0))
      }
    }
  }
  
  /// Accesses `self` as an `ArrayBuffer<Element>`.
  ///
  /// - Requires: `Element.self == elementType`.
  public subscript<Element>(existingElementType _: Type<Element>) -> ArrayBuffer<Element> {
    get {
      ArrayBuffer<Element>(self)!
    }
    
    _modify {
      // TODO: check for spurious ARC traffic
      var me = ArrayBuffer<Element>(self)!
      // Don't retain an additional reference during this mutation, to prevent needless CoW.
      self.storage = nil
      yield &me
      self.storage = .init(me.storage)
    }
  }
  
  /// Accesses `self` as an `ArrayBuffer<Element>` unsafely assuming `Element.self == elementType`.
  ///
  /// - Requires: `Element.self == elementType`.
  public subscript<Element>(unsafelyAssumingElementType _: Type<Element>) -> ArrayBuffer<Element> {
    get {
      ArrayBuffer<Element>(unsafelyDowncasting: self)
    }
    _modify {
      // TODO: check for spurious ARC traffic
      var me = ArrayBuffer<Element>(unsafelyDowncasting: self)
      // Don't retain an additional reference during this mutation, to prevent needless CoW.
      self.storage = nil
      yield &me
      self.storage = .init(me.storage)
    }
  }
}

extension Type {
  static var id: TypeID { TypeID(T.self) }
}
