// TODO: Move all these to Penguin.

import PenguinStructures

extension AnyArrayBuffer {
  /// Appends `x`, returning the index of the appended element.
  ///
  /// - Requires: `self.elementType == T.self`.
  mutating func unsafelyAppend<T>(_ x: T) -> Int {
    self[unsafelyAssumingElementType: Type<T>()].append(x)
  }

  /// Accesses `self` as an `AnyArrayBuffer` with no exposed capabilities
  /// (i.e. `AnyArrayBuffer<AnyObject>`).
  var upcast: AnyArrayBuffer<AnyObject> {
    get { .init(self) }
    _modify {
      var x = AnyArrayBuffer<AnyObject>(unsafelyCasting: self)
      self.storage = nil
      defer { self.storage = x.storage }
      yield &x
    }
  }
}

extension ArrayStorage: CustomDebugStringConvertible {
  public var debugDescription: String { "ArrayStorage(\(Array(self)))" }
}

extension ArrayBuffer: CustomDebugStringConvertible {
  public var debugDescription: String { "ArrayBuffer(\(Array(self)))" }
}

// =================================================================================================
// Standard library extensions

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

extension Dictionary {
  /// Accesses the value associated with a key that is known to exist in `self`.
  ///
  /// - Precondition: self[k] != `nil`.
  subscript(existingKey k: Key) -> Value {
    get { self[k]! }
    _modify {
      yield &values[index(forKey: k)!]
    }
  }

  /// Invokes `update` to mutate each stored `Value`, passing its corresponding `Key` and the
  /// `Index` in `self` where that `(Key, Value)` pair is stored.
  mutating func updateValues(_ update: (Key, inout Value, Index) throws -> Void) rethrows {
    var i = startIndex
    while i != endIndex {
      try update(self[i].key, &values[i], i)
      formIndex(after: &i)
    }
  }
}

extension MutableCollection {
  mutating func writePrefix<I: IteratorProtocol>(from source: inout I)
    -> (writtenCount: Int, unwrittenStart: Index)
    where I.Element == Element
  {
    var writtenCount = 0
    var unwrittenStart = startIndex
    while unwrittenStart != endIndex {
      guard let x = source.next() else { break }
      self[unwrittenStart] = x
      self.formIndex(after: &unwrittenStart)
      writtenCount += 1
    }
    return (writtenCount, unwrittenStart)
  }

  mutating func writePrefix<Source: Collection>(from source: Source)
    -> (writtenCount: Int, unwrittenStart: Index, unreadStart: Source.Index)
    where Source.Element == Element
  {
    var writtenCount = 0
    var unwrittenStart = startIndex
    var unreadStart = source.startIndex
    while unwrittenStart != endIndex && unreadStart != source.endIndex {
      self[unwrittenStart] = source[unreadStart]
      self.formIndex(after: &unwrittenStart)
      source.formIndex(after: &unreadStart)
      writtenCount += 1
    }
    return (writtenCount, unwrittenStart, unreadStart)
  }

  @discardableResult
  mutating func assign<Source: Sequence>(_ sourceElements: Source) -> Int
    where Source.Element == Element
  {
    var stream = sourceElements.makeIterator()
    let (count, unwritten) = writePrefix(from: &stream)
    precondition(unwritten == endIndex, "source too short")
    precondition(stream.next() == nil, "source too long")
    return count
  }
  
  @discardableResult
  mutating func assign<Source: Collection>(_ sourceElements: Source) -> Int
    where Source.Element == Element
  {
    let (writtenCount, unwritten, unread) = writePrefix(from: sourceElements)
    precondition(unwritten == endIndex, "source too short")
    precondition(unread == sourceElements.endIndex, "source too long")
    return writtenCount
  }
}

extension Collection {
  func index(atOffset n: Int) -> Index {
    index(startIndex, offsetBy: n)
  }
}
