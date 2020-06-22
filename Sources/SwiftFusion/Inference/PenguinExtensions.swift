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

/// An `AnyArrayBuffer` dispatcher for elements that are not statically known to support any
/// operations.
// We can't use `AnyObject` for this because the bitcast at [1] fails with "Swift runtime failure:
// Can't unsafeBitCast between types of different sizes" when trying to cast an `AnyArrayBuffer`
// of some other dispatch to an `AnyArrayBuffer<AnyObject>`.
//
// [1] https://github.com/saeta/penguin/blob/47980eb5630ebfd7bc2a88a4a7a60a050dd01b0a/Sources/PenguinStructures/AnyArrayBuffer.swift#L64
class AnyElementDispatch {}

/// A type-erased array that is not statically known to support any operations.
typealias AnyElementArrayBuffer = AnyArrayBuffer<AnyElementDispatch>

extension AnyArrayBuffer where Dispatch == AnyElementDispatch {
  /// Creates an instance from a typed buffer of `Element`
  init<Element>(_ src: ArrayBuffer<Element>) {
    self.init(
      storage: src.storage,
      dispatch: AnyElementDispatch.self)
  }
}

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
