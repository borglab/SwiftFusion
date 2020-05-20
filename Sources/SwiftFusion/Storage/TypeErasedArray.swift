internal struct ArrayHeader {
  var count: Int
  var capacity: Int
}

/// Base class for contiguous storage of homogeneous elements.
class ContiguousStorageBase {
  typealias Implementation = AnyContiguousStorageImplementation
  var implementation: Implementation {
    fatalError("implement me!")
  }
  
  /// Appends the instance of the concrete element type whose address is `p`,
  /// returning the index of the appended element, or `nil` if there was
  /// insufficient capacity remaining
  final func appendValue(at p: UnsafeRawPointer) -> Int? {
    implementation.appendValue_(at: p)
  }

  /// Invokes `body` with the memory occupied by initialized elements.
  final func withUnsafeMutableRawBufferPointer<R>(
    _ body: (inout UnsafeMutableRawBufferPointer)->R
  ) -> R {
    implementation.withUnsafeMutableRawBufferPointer_(body)
  }

  deinit { implementation.deinitialize() }
}

protocol AnyContiguousStorageImplementation: ContiguousStorageBase {
  /// Appends the instance of the concrete element type whose address is `p`,
  /// returning the index of the appended element, or `nil` if there was
  /// insufficient capacity remaining
  func appendValue_(at p: UnsafeRawPointer) -> Int?
  
  /// Invokes `body` with the memory occupied by initialized elements.
  func withUnsafeMutableRawBufferPointer_<R>(
    _ body: (inout UnsafeMutableRawBufferPointer)->R
  ) -> R

  /// Deinitialize stored data
  func deinitialize()
}

protocol ContiguousStorageImplementation: AnyContiguousStorageImplementation {
  associatedtype Element
}

/// APIs that depend on the `Element` type.
extension ContiguousStorageImplementation {
  internal typealias Manager = ManagedBufferPointer<ArrayHeader, Element>
  internal var managed: Manager { Manager(unsafeBufferObject: self) }

  private var header: ArrayHeader {
    _read {
      defer { _fixLifetime(self) }
      let h = managed.withUnsafeMutablePointerToHeader { $0 }
      yield h[0]
    }
    _modify {
      defer { _fixLifetime(self) }
      let h = managed.withUnsafeMutablePointerToHeader { $0 }
      yield &h[0]
    }
  }

  internal var count: Int {
    _read { yield header.count }
    _modify {
      defer { _fixLifetime(self) }
      yield &header.count
    }
  }

  internal var capacity: Int {
    _read { yield header.capacity }
    _modify {
      defer { _fixLifetime(self) }
      yield &header.capacity
    }
  }

  /// Appends `x`, returning the index of the appended element, or `nil` if
  /// there was insufficient capacity remaining.
  func append(_ x: Element) -> Int? {
    let r = count
    if r == capacity { return nil }
    managed.withUnsafeMutablePointers { h, e in
      (e + r).initialize(to: x)
      h[0].count = r + 1
    }
    return r
  }

  /// Invokes `body` with the memory occupied by initialized elements.
  func withUnsafeMutableBufferPointer<R>(
    _ body: (inout UnsafeMutableBufferPointer<Element>) -> R
  ) -> R {
    managed.withUnsafeMutablePointers { h, e in
      var b = UnsafeMutableBufferPointer(start: e, count: h[0].count)
      return body(&b)
    }
  }

  /// Returns an empty instance with `capacity` at least `minimumCapacity`.
  static func create(minimumCapacity: Int) -> Self {
    unsafeDowncast(
      Manager(bufferClass: Self.self, minimumCapacity: minimumCapacity) {
        buffer, getCapacity in 
        ArrayHeader(count: 0, capacity: getCapacity(buffer))
      }.buffer,
      to: Self.self)
  }
}

/// Implementation of `AnyContiguousStorageImplementation` requirements
extension ContiguousStorageImplementation {
  /// Appends the instance of the concrete element type whose address is `p`,
  /// returning the index of the appended element, or `nil` if there was
  /// insufficient capacity remaining
  func appendValue_(at p: UnsafeRawPointer) -> Int? {
    append(p.assumingMemoryBound(to: Element.self)[0])
  }

  /// Invokes `body` with the memory occupied by initialized elements.
  func withUnsafeMutableRawBufferPointer_<R>(
    _ body: (inout UnsafeMutableRawBufferPointer)->R
  ) -> R {
    withUnsafeMutableBufferPointer {
      var b = UnsafeMutableRawBufferPointer($0)
      return body(&b)
    }
  }

  /// Deinitialize stored data. Models should call this from their `deinit`.
  func deinitialize() {
    self.managed.withUnsafeMutablePointers { h, e in
      e.deinitialize(count: h[0].count)
      h.deinitialize(count: 1)
    }
  }
}


final class ContiguousStorage<Element>:
  ContiguousStorageBase, ContiguousStorageImplementation
{
  override var implementation: AnyContiguousStorageImplementation { self }
}
