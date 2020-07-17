// TODO: Move all these to Penguin.

import PenguinStructures

extension AnyArrayBuffer {
  /// Appends `x`, returning the index of the appended element.
  ///
  /// - Requires: `self.elementType == T.self`.
  mutating func unsafelyAppend<T>(_ x: T) -> Int {
    unsafelyMutate(assumingElementType: Type<T>()) { $0.append(x) }
  }
}

/// A type-erased array that is not statically known to support any operations.
typealias AnyElementArrayBuffer = AnyArrayBuffer<AnyObject>

extension ArrayBuffer {
  /// Creates an instance with the given `count`, and capacity at least
  /// `minimumCapacity`, and elements initialized by `initializeElements`,
  /// which is passed the address of the (uninitialized) first element.
  ///
  /// - Requires: `initializeElements` initializes exactly `count` contiguous
  ///   elements starting with the address it is passed.
  public init(
    count: Int,
    minimumCapacity: Int = 0,
    initializeElements:
      (_ uninitializedElements: UnsafeMutablePointer<Element>) -> Void
  ) {
    // Work around the fact that `ArrayBuffer.init(storage:)` doesn't exist. We can just set
    // `storage` directly without calling a different initializer first when we move this
    // implementation to penguin.
    self.init(minimumCapacity: 0)
    self.storage = .init(
      count: count, minimumCapacity: minimumCapacity, initializeElements: initializeElements)
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
