/// A collection that can be initialized from any other collection with the same element type.
public protocol InitializableFromCollection: Collection {
  /// Creates `Self`, containing the elements of `source`.
  init<C: Collection>(_ source: C) where C.Element == Element
}

/// An array whose number of elements is known at compile time.
public protocol FixedSizeArray: InitializableFromCollection, MutableCollection {
  /// The number of elements in the array.
  static var count: Int { get }
}

// MARK: - "Generated Code"

public struct Array1<Element>: FixedSizeArray {
  var e0: Element

  public init(_ e0: Element) {
    self.e0 = e0
  }

  public init<C: Collection>(_ source: C) where C.Element == Element {
    precondition(source.count == 1)
    var iterator = source.makeIterator()
    self.e0 = iterator.next()!
    assert(iterator.next() == nil)
  }

  public subscript(index: Int) -> Element {
    get {
      switch index {
      case 0:
        return e0
      default:
        preconditionFailure("index out of range")
      }
    }
    set(newValue) {
      switch index {
      case 0:
        e0 = newValue
      default:
        preconditionFailure("index out of range")
      }
    }
  }

  public func index(after i: Int) -> Int { return i + 1 }
  public var startIndex: Int { 0 }
  public var endIndex: Int { Self.count }
  public static var count: Int { 1 }
}

extension Array1: Equatable where Element: Equatable {}

public struct Array2<Element>: FixedSizeArray {
  var e0, e1: Element

  public init(_ e0: Element, _ e1: Element) {
    self.e0 = e0
    self.e1 = e1
  }

  public init<C: Collection>(_ source: C) where C.Element == Element {
    precondition(source.count == 2)
    var iterator = source.makeIterator()
    self.e0 = iterator.next()!
    self.e1 = iterator.next()!
    assert(iterator.next() == nil)
  }

  public subscript(index: Int) -> Element {
    get {
      switch index {
      case 0:
        return e0
      case 1:
        return e1
      default:
        preconditionFailure("index out of range")
      }
    }
    set(newValue) {
      switch index {
      case 0:
        e0 = newValue
      case 1:
        e1 = newValue
      default:
        preconditionFailure("index out of range")
      }
    }
  }

  public func index(after i: Int) -> Int { return i + 1 }
  public var startIndex: Int { 0 }
  public var endIndex: Int { Self.count }
  public static var count: Int { 2 }
}

extension Array2: Equatable where Element: Equatable {}

public struct Array3<Element>: FixedSizeArray {
  var e0, e1, e2: Element

  public init(_ e0: Element, _ e1: Element, _ e2: Element) {
    self.e0 = e0
    self.e1 = e1
    self.e2 = e2
  }

  public init<C: Collection>(_ source: C) where C.Element == Element {
    precondition(source.count == 2)
    var iterator = source.makeIterator()
    self.e0 = iterator.next()!
    self.e1 = iterator.next()!
    self.e2 = iterator.next()!
    assert(iterator.next() == nil)
  }

  public subscript(index: Int) -> Element {
    get {
      switch index {
      case 0:
        return e0
      case 1:
        return e1
      case 2:
        return e2
      default:
        preconditionFailure("index out of range")
      }
    }
    set(newValue) {
      switch index {
      case 0:
        e0 = newValue
      case 1:
        e1 = newValue
      case 2:
        e2 = newValue
      default:
        preconditionFailure("index out of range")
      }
    }
  }

  public func index(after i: Int) -> Int { return i + 1 }
  public var startIndex: Int { 0 }
  public var endIndex: Int { Self.count }
  public static var count: Int { 3 }
}

extension Array3: Equatable where Element: Equatable {}
