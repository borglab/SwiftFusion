// TODO: Move all these to Penguin.

import PenguinStructures

/// A suitable `Index` type for a flattened view of a collection-of-collections having `OuterIndex`
/// and `InnerIndex` as the `Index` types of the outer and inner collections, respectively.
public struct FlattenedIndex<OuterIndex: Comparable, InnerIndex: Comparable> : Comparable {
  /// The position of `self` in the outer collection
  public var outer: OuterIndex
  
  /// The position of `self` in the inner collection
  public var inner: InnerIndex?

  /// Returns `true` iff `lhs` precedes `rhs`.
  public static func <(lhs: Self, rhs: Self) -> Bool {
    if lhs.outer < rhs.outer { return true }
    if lhs.outer != rhs.outer { return false }
    guard let l1 = lhs.inner else { return false }
    guard let r1 = rhs.inner else { return true }
    return l1 < r1
  }

  /// Creates an instance with the given stored properties.
  private init(outer: OuterIndex, inner: InnerIndex?) {
    self.outer = outer
    self.inner = inner
  }

  /// Returns an instance that represents the first position in the flattened view of `c`, where the
  /// inner collection for each element `e` of `c` is `innerCollection(e)`.
  internal init<OuterCollection: Collection, InnerCollection: Collection>(
    firstValidIn c: OuterCollection, innerCollection: (OuterCollection.Element)->InnerCollection
  )
    where OuterCollection.Index == OuterIndex, InnerCollection.Index == InnerIndex
  {
    if let i = c.firstIndex(where: { !innerCollection($0).isEmpty }) {
      self.init(outer: i, inner: innerCollection(c[i]).startIndex)
    }
    else {
      self.init(endIn: c)
    }
  }

  /// Replaces `self` with the next position in the flattened view of `c`, where the inner
  /// collection for each element `e` of `c` is `innerCollection(e)`.
  internal mutating func formNextValid<OuterCollection: Collection, InnerCollection: Collection>(
    in c: OuterCollection, innerCollection: (OuterCollection.Element)->InnerCollection
  )
    where OuterCollection.Index == OuterIndex, InnerCollection.Index == InnerIndex
  {
    let c1 = innerCollection(c[outer])
    var i = inner!
    c1.formIndex(after: &i)
    inner = i
    if i != c1.endIndex { return }
    self = .init(firstValidIn: c[outer...].dropFirst(), innerCollection: innerCollection)
  }

  /// Replaces `self` with the next position in the flattened view of `c`, where the inner
  /// collection for each element `e` of `c` is `e` itself.
  internal mutating func formNextValid<OuterCollection: Collection>(in c: OuterCollection)
    where OuterCollection.Index == OuterIndex, OuterCollection.Element: Collection,
          OuterCollection.Element.Index == InnerIndex
  {
    formNextValid(in: c) { $0 }
  }

  /// Creates an instance that represents the first position in the flattened view of `c`, where the
  /// inner collection for each element `e` of `c` is `e` itself.
  internal init<C: Collection>(firstValidIn c: C)
    where C.Index == OuterIndex, C.Element: Collection, C.Element.Index == InnerIndex
  {
    self.init(firstValidIn: c) { $0 }
  }
  
  /// Creates an instance that represents the `endIndex` in a flattened view of `c`.
  internal init<C: Collection>(endIn c: C) where C.Index == OuterIndex
  {
    self.init(outer: c.endIndex, inner: nil)
  }

  /// Creates an instance that represents the next position after `i` in a flattened view of `c`,
  /// where the inner collection for each element `e` of `c` is `innerCollection(e)`.
  init<OuterCollection: Collection, InnerCollection: Collection>(
    nextAfter i: Self, in c: OuterCollection,
    innerCollection: (OuterCollection.Element)->InnerCollection
  )
    where OuterCollection.Index == OuterIndex, InnerCollection.Index == InnerIndex
  {
    let c1 = innerCollection(c[i.outer])
    let nextInner = c1.index(after: i.inner!)
    if nextInner != c1.endIndex {
      self.init(outer: i.outer, inner: nextInner)
      return
    }
    let nextOuter = c.index(after: i.outer)
    self.init(firstValidIn: c[nextOuter...], innerCollection: innerCollection)
  }  

  /// Creates an instance that represents the next position after `i` in a flattened view of `c`,
  /// where the inner collection for each element `e` of `c` is `e` itself.
  init<OuterCollection: Collection>(nextAfter i: Self, in c: OuterCollection)
    where OuterCollection.Index == OuterIndex, OuterCollection.Element: Collection,
          OuterCollection.Element.Index == InnerIndex
  {
    self.init(nextAfter: i, in: c) { $0 }
  }  
}

extension AnyArrayBuffer {
  /// `true` iff `self` contains no elements.
  var isEmpty: Bool { count == 0 }
  
  // TODO: retire in favor of subscript downcast + append?
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

import PenguinParallelWithFoundation

//
// Storage for “vtables” implementing operations on type-erased values.
//
// Large tables of Swift functions are expensive if stored inline, and incur ARC and allocation
// costs if stored in classes.  We want to allocate them once, keep them alive forever, and
// reference them with cheap unsafe pointers.
//
// We therefore need a vtable cache.  For thread-safety, we could share a table and use a lock, but
// the easiest answer for now is use a thread-local cache per thread.
typealias TLS = PosixConcurrencyPlatform.ThreadLocalStorage

/// Holds the lookup table for vtables.  TLS facilities require that this be a class type.
fileprivate class VTableCache {
  /// A map from a pair of types (`Wrapper`, `Wrapped`) to the address of a VTable containing the
  /// implementation of `Wrapper` operations in terms of `Wrapped` operations.
  var tables: [Array2<TypeID>: UnsafeRawPointer] = [:]
  init() {}
}

/// Lookup key for the thread-local vtable cache.
fileprivate let vTableCacheKey = TLS.makeKey(for: Type<VTableCache>())

/// Constructs a vtable cache instance for the current thread.
@inline(never)
fileprivate func makeVTableCache() -> VTableCache {
  let r = VTableCache()
  TLS.set(r, for: vTableCacheKey)
  return r
}

/// Returns a pointer to the table corresponding to `tableID`, creating it by invoking `create` if it
/// is not already available.
fileprivate func demandVTable<Table>(_ tableID: Array2<TypeID>, create: ()->Table)
  -> UnsafePointer<Table>
{
  /// Returns a pointer to a globally-allocated copy of the result of `create`, registering it
  /// in `tableCache`.
  @inline(never)
  func makeAndCacheTable(in cache: VTableCache) -> UnsafePointer<Table> {
    let r = UnsafeMutablePointer<Table>.allocate(capacity: 1)
    r.initialize(to: create())
    cache.tables[tableID] = .init(r)
    return .init(r)
  }
  
  let slowPathCache: VTableCache
  
  if let cache = TLS.get(vTableCacheKey) {
    if let r = cache.tables[tableID] { return r.assumingMemoryBound(to: Table.self) }
    slowPathCache = cache
  }
  else {
    slowPathCache = makeVTableCache()
  }
  return makeAndCacheTable(in: slowPathCache)
}

// =================================================================================================

/// “Existential VTable” for requirements that can't be threaded through closures (see
/// https://forums.swift.org/t/pitch-rethrows-unchecked/10078/9).
fileprivate protocol AnyMutableCollectionExistentialDispatchProtocol {
  static func withContiguousStorageIfAvailable(
    _ self_: AnyValue,
    _ body: (UnsafeRawPointer?, Int) throws -> Void
  ) rethrows

  static func withUnsafeMutableBufferPointerIfSupported(
    _ self_: inout AnyValue,
    _ body: (UnsafeMutableRawPointer?, Int) throws -> Void
  ) rethrows

  static func withContiguousMutableStorageIfAvailable(
    _ self_: inout AnyValue,
    _ body: (UnsafeMutableRawPointer?, Int) throws -> Void
  ) rethrows

  static func partition(
    _ self_: inout AnyValue,
    _ resultIndex: UnsafeMutableRawPointer,
    _ belongsInSecondPartition: (UnsafeRawPointer) throws -> Bool
  ) rethrows
}

/// Instance of `AnyMutableCollectionExistentialDispatchProtocol` for an AnyMutableCollection
/// wrapping `C`.
enum AnyMutableCollectionExistentialDispatch<C: MutableCollection>
  : AnyMutableCollectionExistentialDispatchProtocol {
  static func withContiguousStorageIfAvailable(
    _ self_: AnyValue,
    _ body: (UnsafeRawPointer?, Int) throws -> Void
  ) rethrows {
    try self_[unsafelyAssuming: Type<C>()]
      .withContiguousStorageIfAvailable { try body($0.baseAddress, $0.count) }
  }

  static func withUnsafeMutableBufferPointerIfSupported(
    _ self_: inout AnyValue,
    _ body: (UnsafeMutableRawPointer?, Int) throws -> Void
  ) rethrows {
    try self_[unsafelyAssuming: Type<C>()]
      ._withUnsafeMutableBufferPointerIfSupported { try body($0.baseAddress, $0.count) }
  }

  static func withContiguousMutableStorageIfAvailable(
    _ self_: inout AnyValue,
    _ body: (UnsafeMutableRawPointer?, Int) throws -> Void
  ) rethrows {
    try self_[unsafelyAssuming: Type<C>()]
      .withContiguousMutableStorageIfAvailable { try body($0.baseAddress, $0.count) }
  }

  static func partition(
    _ self_: inout AnyValue,
    _ resultIndex: UnsafeMutableRawPointer,
    _ belongsInSecondPartition: (UnsafeRawPointer) throws -> Bool
  ) rethrows {
    let r = try self_[unsafelyAssuming: Type<C>()]
      .partition { try withUnsafePointer(to: $0) { try belongsInSecondPartition($0) } }
    resultIndex.assumingMemoryBound(to: AnyMutableCollection<C.Element>.Index?.self)
      .pointee = .init(r)
  }
}

/// A type-erased mutable collection similar to the standard library's `AnyCollection`.
public struct AnyMutableCollection<Element> {
  typealias Storage = AnyValue
  private var storage: Storage
  private var vtable: UnsafePointer<VTable>
  
  typealias Self_ = Self
  fileprivate struct VTable {
    // Sequence requirements.
    let makeIterator: (Self_) -> Iterator
    let existentialDispatch: AnyMutableCollectionExistentialDispatchProtocol.Type
    
    // Hidden Sequence requirements
    let customContainsEquatableElement: (Self_, Element) -> Bool?
    let copyToContiguousArray: (Self_) -> ContiguousArray<Element>
    let copyContents: (
      Self_,
      _ uninitializedTarget: UnsafeMutableBufferPointer<Element>)
      -> (Iterator, UnsafeMutableBufferPointer<Element>.Index)

    // Collection requirements.
    let startIndex: (Self_) -> Index
    let endIndex: (Self_) -> Index
    let indexAfter: (Self_, Index) -> Index
    let formIndexAfter: (Self_, inout Index) -> Void
    let subscript_get: (Self_, Index) -> Element
    let subscript_set: (inout Self_, Index, Element) -> Void
    let isEmpty: (Self_) -> Bool
    let count: (Self_) -> Int
    let indexOffsetBy: (Self_, Index, Int) -> Index
    let indexOffsetByLimitedBy: (Self_, Index, Int, _ limit: Index) -> Index?
    let distance: (Self_, _ start: Index, _ end: Index) -> Int

    // let subscriptRange_get: (Self_, Range<Index>) -> SubSequence
    // let subscriptRange_set: (inout Self_, Range<Index>, SubSequence) -> Void
    
    // We'd need an efficient AnyCollection in order to implement this better.
    // let indices: (Self_) -> Indices
    
    // Hidden collection requirements
    let customIndexOfEquatableElement: (Self_, _ element: Element) -> Index??
    let customLastIndexOfEquatableElement: (Self_, _ element: Element) -> Index??

    // MutableCollection requirements
    let swapAt: (inout Self_, Index, Index) -> Void
    
  }
  
  init<C: MutableCollection>(_ x: C) where C.Element == Element {
    storage = Storage(x)
    vtable = demandVTable(.init(Type<Self>.id, Type<C>.id)) {
      VTable(
        makeIterator: { Iterator($0.storage[unsafelyAssuming: Type<C>()].makeIterator()) },
        existentialDispatch: AnyMutableCollectionExistentialDispatch<C>.self,
        customContainsEquatableElement: { self_, e in
          self_.storage[unsafelyAssuming: Type<C>()]._customContainsEquatableElement(e)
        },
        
        copyToContiguousArray: { $0.storage[unsafelyAssuming: Type<C>()]._copyToContiguousArray() },
        copyContents: {
          self_, target in
          let (i, j) = self_.storage[unsafelyAssuming: Type<C>()]
            ._copyContents(initializing: target)
          return (Iterator(i), j)
        },
        startIndex: { Index($0.storage[unsafelyAssuming: Type<C>()].startIndex) },
        endIndex: { Index($0.storage[unsafelyAssuming: Type<C>()].endIndex) },
        indexAfter: { self_, index_ in
          Index(
            self_.storage[unsafelyAssuming: Type<C>()]
              .index(after: index_.storage[Type<C.Index>()]))
        },
        formIndexAfter: { self_, index in
          self_.storage[unsafelyAssuming: Type<C>()]
            .formIndex(after: &index.storage[Type<C.Index>()])
        },
        subscript_get: { self_, index in
          self_.storage[unsafelyAssuming: Type<C>()][index.storage[Type<C.Index>()]]
        },
        subscript_set: { self_, index, newValue in
          self_.storage[unsafelyAssuming: Type<C>()][index.storage[Type<C.Index>()]] = newValue
        },
        isEmpty: { $0.storage[unsafelyAssuming: Type<C>()].isEmpty },
        count: { $0.storage[unsafelyAssuming: Type<C>()].count },
        indexOffsetBy: { self_, i, offset in
          Index(
            self_.storage[unsafelyAssuming: Type<C>()]
              .index(i.storage[Type<C.Index>()], offsetBy: offset))
        },
        indexOffsetByLimitedBy: { self_, i, offset, limit in
          self_.storage[unsafelyAssuming: Type<C>()].index(
            i.storage[Type<C.Index>()],
            offsetBy: offset,
            limitedBy: limit.storage[Type<C.Index>()]
          ).map(Index.init)
        },
        distance: { self_, start, end in
          self_.storage[unsafelyAssuming: Type<C>()].distance(
            from: start.storage[Type<C.Index>()], to: end.storage[Type<C.Index>()])
        },
        // Hidden collection requirements
        customIndexOfEquatableElement: { self_, e in
          self_.storage[unsafelyAssuming: Type<C>()]
            ._customIndexOfEquatableElement(e).map { $0.map(Index.init) }
        },
        customLastIndexOfEquatableElement: { self_, e in
          self_.storage[unsafelyAssuming: Type<C>()]._customLastIndexOfEquatableElement(e)
            .map { $0.map(Index.init) }
        },
        // MutableCollection requirements.
        swapAt: { self_, i, j in 
          self_.storage[unsafelyAssuming: Type<C>()]
            .swapAt(i.storage[Type<C.Index>()], j.storage[Type<C.Index>()])
        }
      )
    }
  }
  
  public var asAny: Any { storage.asAny }
}

extension AnyMutableCollection: Sequence {
  public struct Iterator: IteratorProtocol {
    var storage: Storage
    // There's only one operation, so probably no point bothering with a vtable.
    var next_: (inout Self) -> Element?

    init<T: IteratorProtocol>(_ x: T) where T.Element == Element {
      storage = .init(x)
      next_ = { $0.storage[unsafelyAssuming: Type<T>()].next() }
    }

    public mutating func next() -> Element? { next_(&self) }
  }

  public func makeIterator() -> Iterator { vtable[0].makeIterator(self) }

  /// If `self` supports a contiguous storage representation, assumes that representation and
  /// returns the result of passing the storage to `body`; returns `nil` otherwise.
  ///
  /// - Note: this documentation might be wrong: https://bugs.swift.org/browse/SR-13486
  public func withContiguousStorageIfAvailable<R>(
    _ body: (UnsafeBufferPointer<Element>) throws -> R
  ) rethrows -> R? {
    var result: R? = nil
    try vtable[0].existentialDispatch.withContiguousStorageIfAvailable(storage) {
      try result = body(
        UnsafeBufferPointer(start: $0?.assumingMemoryBound(to: Element.self), count: $1))
    }
    return result
  }

  public func _customContainsEquatableElement(_ e: Element) -> Bool? {
    vtable[0].customContainsEquatableElement(self, e)
  }

  public __consuming func _copyToContiguousArray() -> ContiguousArray<Element> {
    vtable[0].copyToContiguousArray(self)
  }
  
  public __consuming func _copyContents(
    initializing target: UnsafeMutableBufferPointer<Element>
  ) -> (Iterator,UnsafeMutableBufferPointer<Element>.Index) {
    vtable[0].copyContents(self, target)
  }
}

extension AnyMutableCollection: MutableCollection {
  public struct Index: Comparable {
    fileprivate var storage: Storage
    // Note: using a vtable here actually made benchmarks slower.  We might want to try again if we
    // expand what the benchmarks are doing.  My hunch is that the cost of copying an Index is less
    // important than the cost of checking for equality.
    fileprivate let less: (Index, Index) -> Bool
    fileprivate let equal: (Index, Index) -> Bool

    fileprivate init<T: Comparable>(_ x: T) {
      storage = .init(x)
      less = {l, r in l.storage[unsafelyAssuming: Type<T>()] < r.storage[Type<T>()]}
      equal = {l, r in l.storage[unsafelyAssuming: Type<T>()] == r.storage[Type<T>()]}
    }

    public static func == (lhs: Index, rhs: Index) -> Bool {
      lhs.equal(lhs, rhs)
    }
    
    public static func < (lhs: Index, rhs: Index) -> Bool {
      lhs.less(lhs, rhs)
    }
  }

  public var startIndex: Index { vtable[0].startIndex(self) }
  
  public var endIndex: Index { vtable[0].endIndex(self) }
  
  public func index(after x: Index) -> Index { vtable[0].indexAfter(self, x) }
  
  public func formIndex(after x: inout Index) { vtable[0].formIndexAfter(self, &x) }
  
  public subscript(i: Index) -> Element {
    get { vtable[0].subscript_get(self, i) }
    set { vtable[0].subscript_set(&self, i, newValue) }
  }
  
  public var isEmpty: Bool { vtable[0].isEmpty(self) }
  public var count: Int { vtable[0].count(self) }

  public func _customIndexOfEquatableElement(_ element: Element) -> Index?? {
    vtable[0].customIndexOfEquatableElement(self, element)
  }

  public func _customLastIndexOfEquatableElement(_ element: Element) -> Index?? {
    vtable[0].customLastIndexOfEquatableElement(self, element)
  }

  /// Returns an index that is the specified distance from the given index.
  public func index(_ i: Index, offsetBy distance: Int) -> Index {
    vtable[0].indexOffsetBy(self, i, distance)
  }

  /// Returns an index that is the specified distance from the given index,
  /// unless that distance is beyond a given limiting index.
  public func index(
    _ i: Index, offsetBy distance: Int, limitedBy limit: Index
  ) -> Index? {
    vtable[0].indexOffsetByLimitedBy(self, i, distance, limit)
  }

  /// Returns the distance between two indices.
  public func distance(from start: Index, to end: Index) -> Int {
    vtable[0].distance(self, start, end)
  }

  public mutating func _withUnsafeMutableBufferPointerIfSupported<R>(
    _ body: (inout UnsafeMutableBufferPointer<Element>) throws -> R
  ) rethrows -> R? {
    var result: R? = nil
    try vtable[0].existentialDispatch.withUnsafeMutableBufferPointerIfSupported(&storage) {
      var buf = UnsafeMutableBufferPointer(
        start: $0?.assumingMemoryBound(to: Element.self), count: $1)
      try result = body(&buf)
    }
    return result
  }

  public mutating func withContiguousMutableStorageIfAvailable<R>(
    _ body: (inout UnsafeMutableBufferPointer<Element>) throws -> R
  ) rethrows -> R? {
    var result: R? = nil
    try vtable[0].existentialDispatch.withContiguousMutableStorageIfAvailable(&storage) {
      var buf = UnsafeMutableBufferPointer(
        start: $0?.assumingMemoryBound(to: Element.self), count: $1)
      try result = body(&buf)
    }
    return result
  }
  
  /// Reorders the elements of the collection such that all the elements
  /// that match the given predicate are after all the elements that don't
  /// match.
  ///
  /// - Complexity: O(*n*), where *n* is the length of the collection.
  public mutating func partition(
    by belongsInSecondPartition: (Element) throws -> Bool
  ) rethrows -> Index {
    var r: Index? = nil
    try vtable[0].existentialDispatch.partition(&storage, &r) { p in
      try belongsInSecondPartition(p.assumingMemoryBound(to: Element.self).pointee)
    }
    return r.unsafelyUnwrapped
  }

  /// Exchanges the values at the specified indices of the collection.
  public mutating func swapAt(_ i: Index, _ j: Index) {
    vtable[0].swapAt(&self, i, j)
  }
}

// =================================================================================================

public struct FlattenedMutableCollection<Base: MutableCollection>
  where Base.Element: MutableCollection
{
  var base: Base
    
  init(_ base: Base) {
    self.base = base
  }
}

/*
extension Sequence {
  func reduceUntil<R>(
    _ initialValue: R, combine: (R, Element) throws -> (R, done: Bool)
  ) rethrows -> R {
    var r = initialValue
    var i = makeIterator()
    while let e = i.next() {
      let (r1, done) = try combine(r, i.next())
      r = r1
      if done { break }
    }
    return r
  }
}
 */

extension FlattenedMutableCollection: Sequence {
  public typealias Element = Base.Element.Element
  
  public struct Iterator: IteratorProtocol {
    var outer: Base.Iterator
    var inner: Base.Element.Iterator?

    init(outer: Base.Iterator, inner: Base.Element.Iterator? = nil) {
      self.outer = outer
      self.inner = inner
    }

    public mutating func next() -> Element? {
      while true {
        if let r = inner?.next() { return r }
        guard let o = outer.next() else { return nil }
        inner = o.makeIterator()
      }
    }
  }

  public func makeIterator() -> Iterator { Iterator(outer: base.makeIterator()) }

  // https://bugs.swift.org/browse/SR-13486 points out that the withContiguousStorageIfAvailable
  // method is dangerously underspecified, so we don't implement it here.
  
  public func _customContainsEquatableElement(_ e: Element) -> Bool? {
    let m = base.lazy.map({ $0._customContainsEquatableElement(e) })
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
      var (inner, r1) = e._copyContents(
        initializing: .init(start: t.baseAddress.map { $0 + r }, count: t.count - r))
      r += r1

      // See if we ran out of target space before reaching the end of the inner collection.  I think
      // this will be rare, so instead of using an O(N) `e.count` and comparing with `r1` in all
      // passes of the outer loop, spend `inner` seeing if we reached the end and reconstitute it in
      // O(N) if not.
      if inner.next() != nil {
        @inline(never)
        func earlyExit() -> (Iterator, UnsafeMutableBufferPointer<Element>.Index) {
          var i = e.makeIterator()
          for _ in 0..<r1 { _ = i.next() }
          return (.init(outer: outer, inner: i), r)
        }
        return earlyExit()
      }
    }
    return (.init(outer: outer, inner: nil), r)
  }
}

extension FlattenedMutableCollection: MutableCollection {
  public typealias Index = FlattenedIndex<Base.Index, Base.Element.Index>
  
  public var startIndex: Index { .init(firstValidIn: base) }
  
  public var endIndex: Index { .init(endIn: base) }
  
  public func index(after x: Index) -> Index { .init(nextAfter: x, in: base) }
  
  public func formIndex(after x: inout Index) { x.formNextValid(in: base) }
  
  public subscript(i: Index) -> Element {
    get { base[i.outer][i.inner!] }
    set { base[i.outer][i.inner!] = newValue }
    _modify { yield &base[i.outer][i.inner!] }
  }
  
  public var isEmpty: Bool { base.allSatisfy { $0.isEmpty } }
  public var count: Int { base.lazy.map(\.count).reduce(0, +) }

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

/// The consecutive `SubSequence`s of a `Base` collection having a given maximum size.
struct Batched<Base: Collection>: Collection {
  public let base: Base
  public let maxBatchSize: Int
  
  init(_ base: Base, maxBatchSize: Int) {
    precondition(maxBatchSize > 0)
    self.base = base
    self.maxBatchSize = maxBatchSize
  }

  struct Iterator: IteratorProtocol {
    let slices: Batched
    let position: Index

    mutating func next() -> Base.SubSequence? {
      position.sliceStart == position.sliceEnd ? nil : slices[position]
    }
  }

  func makeIterator() -> Iterator { .init(slices: self, position: startIndex) }
  
  struct Index: Comparable {
    var sliceStart, sliceEnd: Base.Index
    static func < (lhs: Self, rhs: Self) -> Bool {
      lhs.sliceStart < rhs.sliceStart
    }
  }

  private func next(after i: Base.Index) -> Base.Index {
    let limit = base.endIndex
    return base.index(i, offsetBy: maxBatchSize, limitedBy: limit) ?? limit
  }
  
  var startIndex: Index {
    .init(sliceStart: base.startIndex, sliceEnd: next(after: base.startIndex))
  }
  
  var endIndex: Index {
    .init(sliceStart: base.endIndex, sliceEnd: base.endIndex)
  }

  func index(after i: Index) -> Index {
    .init(sliceStart: i.sliceEnd, sliceEnd: next(after: i.sliceEnd))
  }

  func formIndex(after i: inout Index) {
    (i.sliceStart, i.sliceEnd) = (i.sliceEnd, next(after: i.sliceEnd))
  }

  subscript(i: Index) -> Base.SubSequence { base[i.sliceStart..<i.sliceEnd] }
}

extension Collection {
  func sliced(intoBatchesOfAtMost maxBatchSize: Int) -> Batched<Self> {
    .init(self, maxBatchSize: maxBatchSize)
  }
}
