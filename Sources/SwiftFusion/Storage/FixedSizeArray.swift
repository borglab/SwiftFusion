//******************************************************************************
// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

// The point of this prototype is to prove that we can generate efficient code
// for single-element insertion and deletion on a statically-sized array, with
// stable types for the end products i.e.,
// 
//   type(of: a.removing(at: i).inserting(x, at: j)) == type(of: a)
//
// and
//
//    type(of: a.inserting(x, at: j).removing(at: i)) == type(of: a)


/// Statically-sized nonempty collections of homogeneous elements.
///
/// This protocol should be thought of as an implementation detail of `ArrayN`;
/// it is not generally useful.
public protocol FixedSizeArray : MutableCollection, RandomAccessCollection,
                          CustomStringConvertible where Index == Int
{
  /// Creates an instance containing exactly the elements of `source`.
  ///
  /// Requires: `source.count == c`, where `c` is the capacity of instances.
  init<Source: Collection>(_ source: Source) where Source.Element == Element
  // Note: we don't have a generalization to `Sequence` because we couldn't
  // find an implementation optimizes nearly as well, and in practice
  // `Sequence`'s that are not `Collection`s are extremely rare.

  /// Creates an instance containing the elements of `source` except the one
  /// at `victim`.
  ///
  /// Requires: `source.indices.contains(victim)`
  init(_ source: ArrayN<Self>, removingAt victim: Index)

  /// Returns a fixed-sized collection containing the same elements as `self`,
  /// with `newElement` inserted at the `target` position.
  func inserting(_ newElement: Element, at target: Index) -> ArrayN<Self>
}

/// Default implementation of `CustomStringConvertible` conformance.
public extension FixedSizeArray {
  var description: String { "\(Array(self))"}
  
  @_transparent
  func withUnsafeBufferPointer<R>(
      _ body: (UnsafeBufferPointer<Element>) throws -> R
  ) rethrows -> R {
    let count = self.count
    return try withUnsafePointer(to: self) { p in
      try body(
          UnsafeBufferPointer<Element>(
              start: UnsafeRawPointer(p)
                  .assumingMemoryBound(to: Element.self),
              count: count))
    }
  }

  @_transparent
  mutating func withUnsafeMutableBufferPointer<R>(
      _ body: (UnsafeMutableBufferPointer<Element>) throws -> R
  ) rethrows -> R {
    let count = self.count
    return try withUnsafeMutablePointer(to: &self) { p in
      try body(
          UnsafeMutableBufferPointer<Element>(
              start: UnsafeMutableRawPointer(p)
                  .assumingMemoryBound(to: Element.self),
              count: count))
    }
  }
}

/// A fixed sized collection of 1 element.
public struct Array1<T> : FixedSizeArray {
  /// The value contained in the collection
  private var head: T

  /// Creates an instance containing `head`.
  fileprivate init(head: T) { self.head = head }

  /// Creates an instance containing exactly the elements of `source`.
  ///
  /// Requires: `source.count == 1`
  @inline(__always)
  public init<Source: Collection>(_ source: Source)
      where Source.Element == Element
  {
    head = source.first!
    precondition(
        source.index(after: source.startIndex) == source.endIndex,
        "Too many elements in source"
    )
  }
  
  /// Creates an instance containing the elements of `source` except the one
  /// at `victim`.
  ///
  /// Requires: `source.indices.contains(victim)`
  public init(_ source: ArrayN<Self>, removingAt victimPosition: Index) {
    self.init(head: source[1 &- victimPosition])
  }

  /// Traps with an appropriate message if `i` is out of range for subscript.
  private func check(_ i: Index) {
    precondition(i == 0, "Index out of range.")
  }

  /// Returns a fixed-sized collection containing the same elements as `self`,
  /// with `newElement` inserted at the `target` position.
  public func inserting(_ newElement: Element, at i: Index) -> Array2<T> {
    return i == 1
        ? Array2(head, newElement)
        : (Array2(newElement, head), check(i)).0
  }

  // ======== Collection Requirements ============
  public typealias Index = Int
  
  /// Accesses the element at `i`.
  public subscript(i: Index) -> T {
    get { check(i); return head }
    set { check(i); head = newValue }
  }
  
  /// The position of the first element.
  public var startIndex: Index { 0 }
  
  /// The position just past the last element.
  public var endIndex: Index { 1 }
}

// ----- Standard conditional conformances. -------
extension Array1 : Equatable where T : Equatable {}
extension Array1 : Hashable where T : Hashable {}
extension Array1 : Comparable where Element : Comparable {
  public static func < (l: Self, r: Self) -> Bool { return false }
}

/// A fixed sized collection that stores one more element than `Tail` does.
public struct ArrayN<Tail: FixedSizeArray> : FixedSizeArray {
  private var head: Element
  private var tail: Tail
  
  public typealias Element = Tail.Element

  /// Creates an instance containing exactly the elements of `source`.
  ///
  /// Requires: `source.count == c`, where `c` is the capacity of instances.
  @inline(__always)
  public init<Source: Collection>(_ source: Source)
      where Source.Element == Element
  {
    head = source.first!
    tail = .init(source.dropFirst())
  }

  /// Creates an instance containing `head` followed by the contents of
  /// `tail`.
  // Could be private, but for a test that is using it.
  internal init(head: Element, tail: Tail) {
    self.head = head
    self.tail = tail
  }

  /// Creates an instance containing the elements of `source` except the one at
  /// `victim`.
  ///
  /// Requires: `source.indices.contains(victim)`
  public init(_ source: ArrayN<Self>, removingAt victimPosition: Index) {
    self = victimPosition == 0
        ? source.tail
        : Self(
            head: source.head,
            tail: .init(source.tail, removingAt: victimPosition &- 1))
  }
  
  /// Returns a fixed-sized collection containing the same elements as `self`,
  /// with `newElement` inserted at the `target` position.
  public func inserting(_ newElement: Element, at i: Index) -> ArrayN<Self> {
    if i == 0 { return .init(head: newElement, tail: self) }
    return .init(head: head, tail: tail.inserting(newElement, at: i &- 1))
  }

  /// Returns a fixed-sized collection containing the same elements as `self`,
  /// with `newElement` inserted at the start.
  public func prepending(_ newElement: Element) -> ArrayN<Self> {
    return .init(head: newElement, tail: self)
  }


  /// Returns a fixed-sized collection containing the elements of self
  /// except the one at `victim`.
  ///
  /// Requires: `indices.contains(victim)`
  public func removing(at victim: Index) -> Tail {
    .init(self, removingAt: victim)
  }
  
  // ======== Collection Requirements ============
  /// Returns the element at `i`.
  public subscript(i: Int) -> Element {
    get {
      if i == 0 { return head }
      return tail[i &- 1]
    }
    set {
      if i == 0 { head = newValue }
      else { tail[i &- 1] = newValue }
    }
  }
  
  /// The position of the first element.
  public var startIndex: Int { 0 }
  /// The position just past the last element.
  public var endIndex: Int { tail.endIndex &+ 1 }
}

// ======== Conveniences ============

public typealias Array2<T> = ArrayN<Array1<T>>
public typealias Array3<T> = ArrayN<Array2<T>>
public typealias Array4<T> = ArrayN<Array3<T>>
public typealias Array5<T> = ArrayN<Array4<T>>
public typealias Array6<T> = ArrayN<Array5<T>>
public typealias Array7<T> = ArrayN<Array6<T>>

public extension ArrayN {
  /// Creates `Self([a0, a1])` efficiently.
  init<T>(_ a0: T, _ a1: T) where Tail == Array1<T> {
    head = a0; tail = Array1(head: a1)
  }
  /// Creates `Self([a0, a1, a2])` efficiently.
  init<T>(_ a0: T, _ a1: T, _ a2: T) where Tail == Array2<T> {
    head = a0; tail = Array2(a1, a2)
  }
  /// Creates `Self([a0, a1, a2, a3])` efficiently.
  init<T>(_ a0: T, _ a1: T, _ a2: T, _ a3: T) where Tail == Array3<T> {
    head = a0; tail = Tail(a1, a2, a3)
  }
  /// Creates `Self([a0, a1, a2, a3, a4])` efficiently.
  init<T>(_ a0: T, _ a1: T, _ a2: T, _ a3: T, _ a4: T) where Tail == Array4<T>
  {
    head = a0; tail = Tail(a1, a2, a3, a4)
  }
  /// Creates `Self([a0, a1, a2, a3, a4, a5])` efficiently.
  init<T>(_ a0: T, _ a1: T, _ a2: T, _ a3: T, _ a4: T, _ a5: T)
      where Tail == Array5<T>
  {
    head = a0; tail = Tail(a1, a2, a3, a4, a5)
  }
  /// Creates `Self([a0, a1, a2, a3, a4, a6])` efficiently.
  init<T>(_ a0: T, _ a1: T, _ a2: T, _ a3: T, _ a4: T, _ a5: T, _ a6: T)
      where Tail == Array6<T>
  {
    head = a0; tail = Tail(a1, a2, a3, a4, a5, a6)
  }
}

// ----- Standard conditional conformances. -------
extension ArrayN : Equatable
  where Element : Equatable, Tail : Equatable {}
extension ArrayN : Hashable
  where Element : Hashable, Tail : Hashable {}
extension ArrayN : Comparable
  where Element : Comparable, Tail : Comparable {
  public static func < (l: Self, r: Self) -> Bool {
    l.head < r.head || !(l.head > r.head) && l.tail < r.tail
  }
}
