// TODO: Move all these to Penguin.

import PenguinStructures

public struct FlattenedScalars<Base: MutableCollection>
  where Base.Element: Vector
{
  var base: Base
    
  init(_ base: Base) {
    self.base = base
  }
}

extension FlattenedScalars: Sequence {
  public typealias Element = Base.Element.Scalars.Element
  
  public struct Iterator: IteratorProtocol {
    var outer: Base.Iterator
    var inner: Base.Element.Scalars.Iterator?

    init(outer: Base.Iterator, inner: Base.Element.Scalars.Iterator? = nil) {
      self.outer = outer
      self.inner = inner
    }

    public mutating func next() -> Element? {
      while true {
        if let r = inner?.next() { return r }
        guard let o = outer.next() else { return nil }
        inner = o.scalars.makeIterator()
      }
    }
  }

  public func makeIterator() -> Iterator { Iterator(outer: base.makeIterator()) }

  // https://bugs.swift.org/browse/SR-13486 points out that the withContiguousStorageIfAvailable
  // method is dangerously underspecified, so we don't implement it here.
  
  public func _customContainsEquatableElement(_ e: Element) -> Bool? {
    let m = base.lazy.map({ $0.scalars._customContainsEquatableElement(e) })
    if let x = m.first(where: { $0 != false }) {
      return x
    }
    return false
  }

  public __consuming func _copyContents(
    initializing t: UnsafeMutableBufferPointer<Element>
  ) -> (Iterator, UnsafeMutableBufferPointer<Element>.Index) {
    var r = 0
    var outer = base.makeIterator()

    while r < t.count, let e = outer.next() {
      var (inner, r1) = e.scalars._copyContents(
        initializing: .init(start: t.baseAddress.map { $0 + r }, count: t.count - r))
      r += r1

      // See if we ran out of target space before reaching the end of the inner collection.  I think
      // this will be rare, so instead of using an O(N) `e.count` and comparing with `r1` in all
      // passes of the outer loop, spend `inner` seeing if we reached the end and reconstitute it in
      // O(N) if not.
      if inner.next() != nil {
        @inline(never)
        func earlyExit() -> (Iterator, UnsafeMutableBufferPointer<Element>.Index) {
          var i = e.scalars.makeIterator()
          for _ in 0..<r1 { _ = i.next() }
          return (.init(outer: outer, inner: i), r)
        }
        return earlyExit()
      }
    }
    return (.init(outer: outer, inner: nil), r)
  }
}

extension FlattenedScalars: MutableCollection {
  public typealias Index = FlattenedIndex<Base.Index, Base.Element.Scalars.Index>

  public var startIndex: Index { .init(firstValidIn: base, innerCollection: \.scalars) }
  
  public var endIndex: Index { .init(endIn: base) }
  
  public func index(after x: Index) -> Index {
    .init(nextAfter: x, in: base, innerCollection: \.scalars)
  }
  
  public func formIndex(after x: inout Index) {
    x.formNextValid(in: base, innerCollection: \.scalars)
  }
  
  public subscript(i: Index) -> Element {
    get { base[i.outer].scalars[i.inner!] }
    set { base[i.outer].scalars[i.inner!] = newValue }
    _modify { yield &base[i.outer].scalars[i.inner!] }
  }
  
  public var isEmpty: Bool { base.allSatisfy { $0.scalars.isEmpty } }
  public var count: Int { base.lazy.map(\.scalars.count).reduce(0, +) }

  /* TODO: implement or discard
  func _customIndexOfEquatableElement(_ element: Element) -> Index?? {
    
  }

  func _customLastIndexOfEquatableElement(_ element: Element) -> Index?? {
    
  }
  
  /// Returns an index that is the specified distance from the given index.
  func index(_ i: Index, offsetBy distance: Int) -> Index {
  }

  /// Returns an index that is the specified distance from the given index,
  /// unless that distance is beyond a given limiting index.
  func index(
    _ i: Index, offsetBy distance: Int, limitedBy limit: Index
  ) -> Index? {
    vtable[0].indexOffsetByLimitedBy(self, i, distance, limit)
  }

  /// Returns the distance between two indices.
  func distance(from start: Index, to end: Index) -> Int {
    vtable[0].distance(self, start, end)
  }
  
  // https://bugs.swift.org/browse/SR-13486 points out that these two
  // methods are dangerously underspecified, so we don't implement them here.
  public mutating func _withUnsafeMutableBufferPointerIfSupported<R>(
    _ body: (inout UnsafeMutableBufferPointer<Element>) throws -> R
  ) rethrows -> R? {
  }

  public mutating func withContiguousMutableStorageIfAvailable<R>(
    _ body: (inout UnsafeMutableBufferPointer<Element>) throws -> R
  ) rethrows -> R? {
  }
  
  /// Reorders the elements of the collection such that all the elements
  /// that match the given predicate are after all the elements that don't
  /// match.
  ///
  /// - Complexity: O(*n*), where *n* is the length of the collection.
  public mutating func partition(
    by belongsInSecondPartition: (Element) throws -> Bool
  ) rethrows -> Index {
  }

  /// Exchanges the values at the specified indices of the collection.
  public mutating func swapAt(_ i: Index, _ j: Index) {
  }
   */
}
