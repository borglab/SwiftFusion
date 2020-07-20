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

  /// Accesses `self` as an `AnyArrayBuffer<D>` if `self.dispatch` has dynamic type `D`.
  ///
  /// - Requires `D.self is Dispatch.Type`.
  /// - Writing requires a non-nil written value.
  /// - Where `!(self.dispatch is D)`:
  ///   - reading returns `nil`.
  ///   - writing changes the dispatch type of `self`.
  ///
  public subscript<D>(dispatch _: Type<D>) -> AnyArrayBuffer<D>? {
    get {
      precondition(D.self is Dispatch.Type)
      return AnyArrayBuffer<D>(self)
    }
    
    _modify {
      precondition(D.self is Dispatch.Type)
      
      // TODO: check for spurious ARC traffic
      var me = AnyArrayBuffer<D>(self)
      if me != nil {
        // Don't retain an additional reference during this mutation, to prevent needless CoW.
        self.storage = nil
      }
      yield &me
      if let written = me {
        // This unwrapping will succeed due to the precondition (already checked).
        self = Self(written).unsafelyUnwrapped
      }
      else {
        // FIXME: remove all elements instead.
        fatalError("Can't write nil through subscript(dispatch:)")
      }
    }
  }
}
