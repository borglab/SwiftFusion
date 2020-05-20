/// An identifier of a given abstract value with the value's type attached
///
/// - Parameter Value: the type of value this ID refers to.
/// - Parameter PerTypeID: a type that, given `Value`, identifies a given
///   logical value of that type.
public struct TypedID<Value, PerTypeID: Equatable> {
  /// A specifier of which logical value of type `value` is being identified.
  let perTypeID: PerTypeID

  /// Creates an instance indicating the given logical value of type `Value`.
  init(_ perTypeID: PerTypeID) { self.perTypeID = perTypeID }
}

public typealias SimpleTypedID<Value> = TypedID<Value, Int>

// MARK: - Storage without any operations.

/// An existential type for instances of `HomogeneousStorage` that is agnostic to
/// the element type of the underlying array.
///
/// Any given model is associated with a particular element type.
protocol AnyHomogeneousStorage {
  /// Appends the instance of the concrete element type whose address is `p`,
  /// returning the index of the appended element.
  mutating func appendValue(at p: UnsafeRawPointer) -> Int
  
  /// Returns the address of the `n`th element.
  ///
  /// - Warning: the lifetime of `self` must be maintained while the address is
  ///   used.
  mutating func mutableAddress(ofElement n: Int) -> UnsafeMutableRawPointer

  /// Returns the address of the `n`th element.
  ///
  /// - Warning: the lifetime of `self` must be maintained while the address is
  ///   used.
  func address(ofElement i: Int) -> UnsafeRawPointer
  
  /// Returns a mutable buffer addressing all elements.
  ///
  /// - Warning: the lifetime of `self` must be maintained while the buffer is
  ///   used.
  var mutableBuffer: UnsafeMutableRawBufferPointer { mutating get }

  /// Returns a buffer addressing all elements.
  ///
  /// - Warning: the lifetime of `self` must be maintained while the buffer is
  ///   used.
  var buffer: UnsafeRawBufferPointer { get }

  // MARK: - Factor methods.

  /// Returns the total error of all the factors.
  func error(at values: PackedStorage, _ factorInputs: PackedStorage) -> Double

  /// Accumulates the error vectors of all the factors to `out`.
  /// Precondition: `out` has entries for the error vectors of all the elements.
  func errorVector(at values: PackedStorage, _ out: inout PackedStorage, _ factorInputs: PackedStorage)

  /// Computes linear approximations of the elements at `values` and writes the approximations to
  /// `out`.
  func linearize(at values: PackedStorage, _ out: inout FactorGraph, _ factorInputs: PackedStorage)

  /// Applies all the elements' forward linear maps to `x` and accumulates the results to `y`.
  /// Precondition: `y` has an entry at all the elements' outputs.
  func applyLinearForward(_ x: PackedStorage, _ y: inout PackedStorage, _ factorInputs: PackedStorage)

  /// Applies all the elements' tranpose linear maps to `y` and accumulates the results to `x`.
  /// Precondition: `x` has an entry at all the elements' inputs.
  func applyLinearTranspose(_ y: PackedStorage, _ x: inout PackedStorage, _ factorInputs: PackedStorage)

  // MARK: - Differentiable methods.

  mutating func move(along direction: AnyHomogeneousStorage)

  var zeroTangentVector: (ObjectIdentifier, AnyHomogeneousStorage) { get }

  // MARK: - EuclideanVectorSpace methods.

  func scaled(by scalar: Double) -> AnyHomogeneousStorage

  func plus (_ other: AnyHomogeneousStorage) -> AnyHomogeneousStorage

  var squaredNorm: Double { get }
}

extension AnyHomogeneousStorage {
  func error(at values: PackedStorage, _ factorInputs: PackedStorage) -> Double {
    fatalError("unsupported")
  }

  func errorVector(at values: PackedStorage, _ out: inout PackedStorage, _ factorInputs: PackedStorage) {
    fatalError("unsupported")
  }

  func linearize(at values: PackedStorage, _ out: inout FactorGraph, _ factorInputs: PackedStorage) {
    fatalError("unsupported")
  }

  func applyLinearForward(_ x: PackedStorage, _ y: inout PackedStorage, _ factorInputs: PackedStorage) {
    fatalError("unsupported")
  }

  func applyLinearTranspose(_ y: PackedStorage, _ x: inout PackedStorage, _ factorInputs: PackedStorage) {
    fatalError("unsupported")
  }

  mutating func move(along direction: AnyHomogeneousStorage) {
    fatalError("unsupported")
  }

  var zeroTangentVector: (ObjectIdentifier, AnyHomogeneousStorage) {
    fatalError("unsupported")
  }

  func scaled(by scalar: Double) -> AnyHomogeneousStorage {
    fatalError("unsupported")
  }

  func plus (_ other: AnyHomogeneousStorage) -> AnyHomogeneousStorage {
    fatalError("unsupported")
  }

  var squaredNorm: Double {
    fatalError("unsupported")
  }
}

/// A contiguous buffer of `T` instances that can be handled as an existential.
struct HomogeneousStorage<T>: AnyHomogeneousStorage {
  /// The actual storage for `T` instances.
  var storage: [T] = []
  
  /// Appends the instance of the concrete element type whose address is `p`,
  /// returning the index of the appended element.
  mutating func appendValue(at p: UnsafeRawPointer) -> Int {
    storage.append(p.assumingMemoryBound(to: T.self).pointee)
    return storage.count - 1
  }
  
  /// Returns the address of the `n`th element.
  ///
  /// - Warning: the lifetime of `self` must be maintained while the address is
  ///   used.
  mutating func mutableAddress(ofElement i: Int) -> UnsafeMutableRawPointer {
    withUnsafeMutablePointer(to: &storage[i]) { .init($0) }
  }

  /// Returns the address of the `n`th element.
  ///
  /// - Warning: the lifetime of `self` must be maintained while the address is
  ///   used.
  func address(ofElement i: Int) -> UnsafeRawPointer {
    precondition(i >= 0 && i < storage.count, "index out of range")
    return storage.withUnsafeBufferPointer { .init($0.baseAddress! + i) }
  }

  /// Returns a mutable buffer addressing all elements.
  ///
  /// - Warning: the lifetime of `self` must be maintained while the buffer is
  ///   used.
  var mutableBuffer: UnsafeMutableRawBufferPointer {
    mutating get {
      .init(storage.withUnsafeMutableBufferPointer { $0 })
    }
  }
  
  /// Returns a buffer addressing all elements.
  ///
  /// - Warning: the lifetime of `self` must be maintained while the buffer is
  ///   used.
  var buffer: UnsafeRawBufferPointer {
    get {
      .init(storage.withUnsafeBufferPointer { $0 })
    }
  }
}

struct HomogeneousStorageFactor<T: MyFactor>: AnyHomogeneousStorage {
  var base: HomogeneousStorage<T> = HomogeneousStorage()

  // Delegate implementation to `base`.
  mutating func appendValue(at p: UnsafeRawPointer) -> Int {
    base.appendValue(at: p)
  }
  mutating func mutableAddress(ofElement i: Int) -> UnsafeMutableRawPointer {
    base.mutableAddress(ofElement: i)
  }
  func address(ofElement i: Int) -> UnsafeRawPointer {
    base.address(ofElement: i)
  }
  var mutableBuffer: UnsafeMutableRawBufferPointer {
    mutating get {
      base.mutableBuffer
    }
  }
  var buffer: UnsafeRawBufferPointer {
    get {
      base.buffer
    }
  }

  private func theseFactorInputs(_ factorInputs: PackedStorage) -> HomogeneousStorage<T.Input.InputID> {
    guard let factorInputs =
      factorInputs.homogeneousStorage[ObjectIdentifier(T.self)] as?
      HomogeneousStorage<T.Input.InputID>
    else {
      fatalError("wrong factor input type")
    }
    return factorInputs
  }

  /// Returns the total error of all the factors.
  func error(at values: PackedStorage, _ factorInputs: PackedStorage) -> Double {
    let factorInputs = theseFactorInputs(factorInputs)
    var result = Double(0)
    for index in base.storage.indices {
      let factor = base.storage[index]
      let factorInput = factorInputs.storage[index]
      result += factor.error(at: T.Input.get(values, factorInput))
    }
    return result
  }

  /// Accumulates the error vectors of all the factors to `out`.
  /// Precondition: `out` has entries for the error vectors of all the elements.
  func errorVector(
    at values: PackedStorage,
    _ out: inout PackedStorage,
    _ factorInputs: PackedStorage
  ) {
    let factorInputs = theseFactorInputs(factorInputs)
    let outBuffer = out.homogeneousStorage[ObjectIdentifier(T.self)]!
      .mutableBuffer.bindMemory(to: T.ErrorVector.self)
    for index in base.storage.indices {
      let factor = base.storage[index]
      let factorInput = factorInputs.storage[index]
      outBuffer[index] += factor.errorVector(at: T.Input.get(values, factorInput))
    }
  }

  /// Computes linear approximations of the elements at `values` and writes the approximations to
  /// `out`.
  func linearize(
    at values: PackedStorage,
    _ out: inout FactorGraph,
    _ factorInputs: PackedStorage
  ) {
    let factorInputs = theseFactorInputs(factorInputs)
    for index in base.storage.indices {
      let factor = base.storage[index]
      let factorInput = factorInputs.storage[index]
      out += (
        factor.linearized(id: factorInput),
        factor.linearized(at: T.Input.get(values, factorInput))
      )
    }
  }

  /// Applies all the elements' forward linear maps to `x` and accumulates the results to `y`.
  /// Precondition: `y` has an entry at all the elements' outputs.
  func applyLinearForward(_ x: PackedStorage, _ y: inout PackedStorage, _ factorInputs: PackedStorage) {
    let factorInputs = theseFactorInputs(factorInputs)
    let yBuffer = y.homogeneousStorage[ObjectIdentifier(T.self)]!
      .mutableBuffer.bindMemory(to: T.ErrorVector.self)
    for index in base.storage.indices {
      let factor = base.storage[index]
      let factorInput = factorInputs.storage[index]
      yBuffer[index] += factor.applyLinearForward(T.Input.get(x, factorInput))
    }
  }

  /// Applies all the elements' tranpose linear maps to `y` and accumulates the results to `x`.
  /// Precondition: `x` has an entry at all the elements' inputs.
  func applyLinearTranspose(_ y: PackedStorage, _ x: inout PackedStorage, _ factorInputs: PackedStorage) {
    let factorInputs = theseFactorInputs(factorInputs)
    defer { _fixLifetime(y) }
    let yBuffer = y.homogeneousStorage[ObjectIdentifier(T.self)]!
      .buffer.bindMemory(to: T.ErrorVector.self)
    for index in base.storage.indices {
      let factor = base.storage[index]
      let factorInput = factorInputs.storage[index]
      var xProjected = T.Input.get(x, factorInput)
      T.Input.add(&xProjected, factor.applyLinearTranspose(yBuffer[index]))
      T.Input.set(&x, factorInput, to: xProjected)
    }
  }
}

struct HomogeneousStorageDifferentiable<T: Differentiable>: AnyHomogeneousStorage where T.TangentVector: EuclideanVectorSpace {
  var base: HomogeneousStorage<T> = HomogeneousStorage()

  // Delegate implementation to `base`.
  mutating func appendValue(at p: UnsafeRawPointer) -> Int {
    base.appendValue(at: p)
  }
  mutating func mutableAddress(ofElement i: Int) -> UnsafeMutableRawPointer {
    base.mutableAddress(ofElement: i)
  }
  func address(ofElement i: Int) -> UnsafeRawPointer {
    base.address(ofElement: i)
  }
  var mutableBuffer: UnsafeMutableRawBufferPointer {
    mutating get {
      base.mutableBuffer
    }
  }
  var buffer: UnsafeRawBufferPointer {
    get {
      base.buffer
    }
  }

  mutating func move(along direction: AnyHomogeneousStorage) {
    if let direction = direction as? HomogeneousStorageEuclideanVector<T.TangentVector> {
      precondition(base.storage.count == direction.base.storage.count)
      for index in base.storage.indices {
        base.storage[index].move(along: direction.base.storage[index])
      }
      return
    }
    if let direction = direction as? HomogeneousStorageDifferentiable<T.TangentVector> {
      precondition(base.storage.count == direction.base.storage.count)
      for index in base.storage.indices {
        base.storage[index].move(along: direction.base.storage[index])
      }
      return
    }
    fatalError("bad type")
  }

  var zeroTangentVector: (ObjectIdentifier, AnyHomogeneousStorage) {
    return (
      ObjectIdentifier(T.TangentVector.self),
      HomogeneousStorageEuclideanVector<T.TangentVector>(
        base: HomogeneousStorage(storage: base.storage.map { $0.zeroTangentVector } )
      )
    )
  }
}

struct HomogeneousStorageEuclideanVector<T: EuclideanVectorSpace>: AnyHomogeneousStorage {
  var base: HomogeneousStorage<T> = HomogeneousStorage()

  // Delegate implementation to `base`.
  mutating func appendValue(at p: UnsafeRawPointer) -> Int {
    base.appendValue(at: p)
  }
  mutating func mutableAddress(ofElement i: Int) -> UnsafeMutableRawPointer {
    base.mutableAddress(ofElement: i)
  }
  func address(ofElement i: Int) -> UnsafeRawPointer {
    base.address(ofElement: i)
  }
  var mutableBuffer: UnsafeMutableRawBufferPointer {
    mutating get {
      base.mutableBuffer
    }
  }
  var buffer: UnsafeRawBufferPointer {
    get {
      base.buffer
    }
  }

  mutating func move(along direction: AnyHomogeneousStorage) {
    guard let direction = direction as? HomogeneousStorageEuclideanVector<T.TangentVector> else {
      fatalError("bad type")
    }
    precondition(base.storage.count == direction.base.storage.count)
    for index in base.storage.indices {
      base.storage[index].move(along: direction.base.storage[index])
    }
  }

  var zeroTangentVector: AnyHomogeneousStorage {
    return HomogeneousStorageEuclideanVector<T.TangentVector>(
      base: HomogeneousStorage(storage: base.storage.map { $0.zeroTangentVector } )
    )
  }

  func scaled(by scalar: Double) -> AnyHomogeneousStorage {
    return HomogeneousStorageEuclideanVector<T>(
      base: HomogeneousStorage(storage: base.storage.map { $0.scaled(by: scalar) })
    )
  }

  func plus (_ other: AnyHomogeneousStorage) -> AnyHomogeneousStorage {
    guard let other = other as? HomogeneousStorageEuclideanVector<T> else {
      fatalError("bad type")
    }
    precondition(base.storage.count == other.base.storage.count)
    return HomogeneousStorageEuclideanVector<T>(
      base: HomogeneousStorage(
        storage: zip(base.storage, other.base.storage).map { zipped in zipped.0 + zipped.1 }
      )
    )
  }

  var squaredNorm: Double {
    return base.storage.reduce(0) { result, element in result + element.squaredNorm }
  }

}

/// Storage for instances of heterogeneous type in contiguous homogeneous
/// memory.
public struct PackedStorage {
  public init() {  }
  
  /// A mapping from type onto the homogeneous storage for that type.
  var homogeneousStorage: [ObjectIdentifier: AnyHomogeneousStorage] = [:]

  /// A mapping from type onto the tangent vector for that type.
  var tangentVector: [ObjectIdentifier: ObjectIdentifier] = [:]

  /// Stores `initialValue` as a new logical value and returns its ID.
  public mutating func store<T>(_ initialValue: T, tag: ObjectIdentifier = ObjectIdentifier(T.self)) -> TypedID<T, Int> {
    .init(
      withUnsafePointer(to: initialValue) { p in
        homogeneousStorage[
          tag, default: HomogeneousStorage<T>()]
          .appendValue(at: .init(p))
      }
    )
  }

  /// Stores `initialValue` as a new logical value and returns its ID.
  public mutating func storeFactor<T: MyFactor>(_ initialValue: T, tag: ObjectIdentifier = ObjectIdentifier(T.self)) -> TypedID<T, Int> {
    .init(
      withUnsafePointer(to: initialValue) { p in
        homogeneousStorage[
          tag, default: HomogeneousStorageFactor<T>()]
          .appendValue(at: .init(p))
      }
    )
  }

  /// Stores `initialValue` as a new logical value and returns its ID.
  public mutating func storeDifferentiable<T: Differentiable>(_ initialValue: T, tag: ObjectIdentifier = ObjectIdentifier(T.self)) -> TypedID<T, Int>
    where T.TangentVector: EuclideanVectorSpace
  {
    tangentVector[ObjectIdentifier(T.self)] = ObjectIdentifier(T.TangentVector.self)
    return .init(
      withUnsafePointer(to: initialValue) { p in
        homogeneousStorage[
          tag, default: HomogeneousStorageDifferentiable<T>()]
          .appendValue(at: .init(p))
      }
    )
  }

  /// Stores `initialValue` as a new logical value and returns its ID.
  public mutating func storeEuclideanVector<T: EuclideanVectorSpace>(_ initialValue: T, tag: ObjectIdentifier = ObjectIdentifier(T.self)) -> TypedID<T, Int> {
    tangentVector[ObjectIdentifier(T.self)] = ObjectIdentifier(T.TangentVector.self)
    return .init(
      withUnsafePointer(to: initialValue) { p in
        homogeneousStorage[
          tag, default: HomogeneousStorageEuclideanVector<T>()]
          .appendValue(at: .init(p))
      }
    )
  }

  /// Traps with an error indicating that an attempt was made to access a stored
  /// variable of a type that is not represented in `self`.
  private static var noSuchType: AnyHomogeneousStorage {
    fatalError("No such stored variable type")
  }

  /// Accesses the stored value with the given ID.
  public subscript<T>(id: TypedID<T, Int>, tag: ObjectIdentifier = ObjectIdentifier(T.self)) -> T {
    _read {
      yield homogeneousStorage[
        tag, default: Self.noSuchType]
        .address(ofElement: id.perTypeID)
        .assumingMemoryBound(to: T.self).pointee
    }
    _modify {
      defer { _fixLifetime(self) }
      yield &homogeneousStorage[
        tag, default: Self.noSuchType
      ]
        .mutableAddress(ofElement: id.perTypeID)
        .assumingMemoryBound(to: T.self).pointee
    }
  }

  // MARK: - Factor methods.

  /// Returns the total error of all the factors.
  func error(
    at values: PackedStorage,
    _ factorInputs: PackedStorage
  ) -> Double {
    var result = Double(0)
    for factors in homogeneousStorage.values {
      result += factors.error(at: values, factorInputs)
    }
    return result
  }

  /// Accumulates the error vectors of all the factors to `out`.
  /// Precondition: `out` has entries for the error vectors of all the elements.
  func errorVector(at values: PackedStorage, _ out: inout PackedStorage, _ factorInputs: PackedStorage) {
    for factors in homogeneousStorage.values {
      factors.errorVector(at: values, &out, factorInputs)
    }
  }

  /// Computes linear approximations of the elements at `values` and writes the approximations to
  /// `out`.
  func linearize(at values: PackedStorage, _ out: inout FactorGraph, _ factorInputs: PackedStorage) {
    for factors in homogeneousStorage.values {
      factors.linearize(at: values, &out, factorInputs)
    }
  }

  /// Applies all the elements' forward linear maps to `x` and accumulates the results to `y`.
  /// Precondition: `y` has an entry at all the elements' outputs.
  func applyLinearForward(_ x: PackedStorage, _ y: inout PackedStorage, _ factorInputs: PackedStorage) {
    for factors in homogeneousStorage.values {
      factors.applyLinearForward(x, &y, factorInputs)
    }
  }

  /// Applies all the elements' tranpose linear maps to `y` and accumulates the results to `x`.
  /// Precondition: `x` has an entry at all the elements' inputs.
  func applyLinearTranspose(_ y: PackedStorage, _ x: inout PackedStorage, _ factorInputs: PackedStorage) {
    for factors in homogeneousStorage.values {
      factors.applyLinearTranspose(y, &x, factorInputs)
    }
  }

  // MARK: - Differentiable methods.

  public mutating func move(along direction: PackedStorage) {
    for key in homogeneousStorage.keys {
      homogeneousStorage[key]!.move(along: direction.homogeneousStorage[tangentVector[key]!]!)
    }
  }

  public var zeroTangentVector: PackedStorage {
    var result = PackedStorage()
    for value in homogeneousStorage.values {
      let (objectID, zero) = value.zeroTangentVector
      result.homogeneousStorage[objectID] = zero
    }
    return result
  }

  // MARK: - EuclideanVectorSpace methods.

  public func scaled(by scalar: Double) -> PackedStorage {
    var result = PackedStorage()
    for (objectID, value) in homogeneousStorage {
      result.homogeneousStorage[objectID] = value.scaled(by: scalar)
    }
    return result
  }

  public static func + (_ lhs: PackedStorage, _ rhs: PackedStorage) -> PackedStorage {
    var result = PackedStorage()
    for (objectID, value) in lhs.homogeneousStorage {
      result.homogeneousStorage[objectID] = value.plus(rhs.homogeneousStorage[objectID]!)
    }
    return result
  }

  public var squaredNorm: Double {
    var result = Double(0)
    for value in homogeneousStorage.values {
      result += value.squaredNorm
    }
    return result
  }
}

extension PackedStorage: Equatable {
  public static func == (_ lhs: PackedStorage, _ rhs: PackedStorage) -> Bool {
    fatalError("unsupported")
  }
}

extension PackedStorage: EuclideanVectorSpace {
  public typealias TangentVector = PackedStorage
  public static func += (_ lhs: inout PackedStorage, _ rhs: PackedStorage) {
    fatalError("unsupported")
  }
  public static func - (_ lhs: PackedStorage, _ rhs: PackedStorage) -> PackedStorage {
    fatalError("unsupported")
  }
  public static func -= (_ lhs: inout PackedStorage, _ rhs: PackedStorage) {
    fatalError("unsupported")
  }
  public static var zero: PackedStorage {
    fatalError("unsupported")
  }
  public mutating func add(_ x: Double) {
    fatalError("unsupported")
  }
  public func adding(_ x: Double) -> PackedStorage {
    fatalError("unsupported")
  }
  public mutating func subtract(_ x: Double) {
    fatalError("unsupported")
  }
  public func subtracting(_ x: Double) -> PackedStorage {
    fatalError("unsupported")
  }
  public mutating func scale(by scalar: Double) {
    fatalError("unsupported")
  }
}

public protocol FactorInputProtocol {
  associatedtype InputID
  static func get(_ storage: PackedStorage, _ id: InputID) -> Self
  static func set(_ storage: inout PackedStorage, _ id: InputID, to value: Self)

  // Hack type-erased vector addition.
  static func add(_ x: inout Self, _ y: Self)
}

extension FactorInputProtocol {
  static func add(_ x: inout Self, _ y: Self) {
    fatalError("unsupported")
  }
}

struct FactorInput1<A>: FactorInputProtocol {
  typealias InputID = SimpleTypedID<A>
  static func get(_ storage: PackedStorage, _ id: InputID) -> Self {
    return Self(value1: storage[id])
  }
  static func set(_ storage: inout PackedStorage, _ id: InputID, to value: Self) {
    storage[id] = value.value1
  }
  var value1: A
}

extension FactorInput1: Differentiable where A: Differentiable, A.TangentVector: EuclideanVectorSpace & VectorConvertible {
  typealias TangentVector = FactorVectorInput1<A.TangentVector>
  mutating func move(along direction: TangentVector) {
    value1.move(along: direction.value1)
  }
}

struct FactorInput2<A, B>: FactorInputProtocol {
  typealias InputID = (SimpleTypedID<A>, SimpleTypedID<B>)
  static func get(_ storage: PackedStorage, _ id: InputID) -> Self {
    return Self(value1: storage[id.0], value2: storage[id.1])
  }
  static func set(_ storage: inout PackedStorage, _ id: InputID, to value: Self) {
    storage[id.0] = value.value1
    storage[id.1] = value.value2
  }
  var value1: A
  var value2: B
}

extension FactorInput2: Differentiable
  where A: Differentiable, A.TangentVector: FixedDimensionVector & EuclideanVectorSpace & VectorConvertible,
    B: Differentiable, B.TangentVector: EuclideanVectorSpace & VectorConvertible
{
  typealias TangentVector = FactorVectorInput2<A.TangentVector, B.TangentVector>
  mutating func move(along direction: TangentVector) {
    value1.move(along: direction.value1)
    value2.move(along: direction.value2)
  }
}

struct FactorVectorInput1<A: EuclideanVectorSpace & VectorConvertible>:
  FactorInputProtocol & EuclideanVectorSpace & VectorConvertible
{
  typealias InputID = SimpleTypedID<A>
  static func get(_ storage: PackedStorage, _ id: InputID) -> Self {
    return FactorVectorInput1(value1: storage[id])
  }
  static func set(_ storage: inout PackedStorage, _ id: InputID, to value: Self) {
    storage[id] = value.value1
  }
  var value1: A
  init(value1: A) {
    self.value1 = value1
  }

  var squaredNorm: Double {
    return value1.squaredNorm
  }

  @differentiable
  init(_ vector: Vector) {
    self.value1 = A(vector)
  }
  @differentiable
  var vector: Vector {
    return self.value1.vector
  }

  static func add(_ x: inout Self, _ y: Self) {
    x += y
  }
}

struct FactorVectorInput2<
  A: FixedDimensionVector & EuclideanVectorSpace & VectorConvertible,
  B: EuclideanVectorSpace & VectorConvertible
>: FactorInputProtocol & EuclideanVectorSpace & VectorConvertible {
  typealias InputID = (SimpleTypedID<A>, SimpleTypedID<B>)
  static func get(_ storage: PackedStorage, _ id: InputID) -> Self {
    return FactorVectorInput2(value1: storage[id.0], value2: storage[id.1])
  }
  static func set(_ storage: inout PackedStorage, _ id: InputID, to value: Self) {
    storage[id.0] = value.value1
    storage[id.1] = value.value2
  }
  var value1: A
  var value2: B
  init(value1: A, value2: B) {
    self.value1 = value1
    self.value2 = value2
  }

  var squaredNorm: Double {
    return value1.squaredNorm + value2.squaredNorm
  }

  @differentiable
  init(_ vector: Vector) {
    self.value1 = withoutDerivative(at: A(Vector(Array(vector.scalars[0..<A.dimension]))))
    self.value2 = withoutDerivative(at: B(Vector(Array(vector.scalars[A.dimension...]))))
  }
  @differentiable
  var vector: Vector {
    return Vector(self.value1.vector.scalars + self.value2.vector.scalars)
  }

  static func add(_ x: inout Self, _ y: Self) {
    x += y
  }
}

// extension FactorInput1: EuclideanVectorSpace where A: EuclideanVectorSpace {
//   var squaredNorm: Double {
//     return value1.squaredNorm
//   }
// }
// 
// extension FactorInput1: VectorConvertible where A: VectorConvertible {
//   @differentiable
//   init(_ vector: Vector) {
//     self.value1 = A(vector)
//   }
// 
//   @differentiable
//   var vector: Vector {
//     return self.value1.vector
//   }
// }

//struct FactorInput2<A, B>: FactorInputProtocol {
//  typealias InputID = (SimpleTypedID<A>, SimpleTypedID<B>)
//  static func index(_ storage: PackedStorage, _ id: InputID) -> Self {
//    return Self(value1: storage[id.0], value2: storage[id.1])
//  }
//  static func set(_ storage: PackedStorage, _ id: InputID, to value: Self) {
//    storage[id.0] = value.value1
//    storage[id.1] = value.value2
//  }
//  var value1: A
//  var value2: B
//}
//
//extension FactorInput2: Differentiable where A: Differentiable, B: Differentiable {}
//
//extension FactorInput2: EuclideanVectorSpace where A: EuclideanVectorSpace {
//  var squaredNorm: Double {
//    return value1.squaredNorm + value2.squaredNorm
//  }
//}
//
//extension FactorInput2: VectorConvertible where A: FixedDimensionVector & VectorConvertible, B: VectorConvertible {
//  @differentiable
//  init(_ vector: Vector) {
//    self.value1 = A(Vector(Array(vector[0..<A.dimension])))
//    self.value2 = B(Vector(Array(vector[A.dimension...])))
//  }
//
//  @differentiable
//  var vector: Vector {
//    return Vector(self.value1.vector.scalars + self.value2.vector.scalars)
//  }
//}

/// A factor that can be stored in homogeneous storage.
// TODO: There should be multiple protocols because not all factors support all of the operations.
public protocol MyFactor {
  /// 
  associatedtype Input: FactorInputProtocol

  /// Returns the error of this factor.
  func error(at value: Input) -> Double

  /// 
  associatedtype ErrorVector: EuclideanVectorSpace

  /// Returns the error vector.
  func errorVector(at value: Input) -> ErrorVector

  associatedtype LinearizedFactor: MyFactor

  /// Returns a linear approximation of this factor at `values`.
  // TODO: It would be very nice to specify linearization with respect to a subset of inputs.
  func linearized(at value: Input) -> LinearizedFactor

  func linearized(id: Input.InputID) -> LinearizedFactor.Input.InputID

  /// Applies this Gaussian factor's forward linear map to `x`.
  func applyLinearForward(_ x: Input) -> ErrorVector

  /// Applies this Gaussian factor's transpose linear map to `y`.
  func applyLinearTranspose(_ y: ErrorVector) -> Input
}

extension MyFactor {
  func error(at value: Input) -> Double {
    return errorVector(at: value).squaredNorm
  }

  func applyLinearForward(_ x: Input) -> ErrorVector {
    fatalError("unsupported")
  }

  func applyLinearTranspose(_ y: ErrorVector) -> Input {
    fatalError("unsupported")
  }
}
