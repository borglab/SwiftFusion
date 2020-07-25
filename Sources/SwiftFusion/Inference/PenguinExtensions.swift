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
