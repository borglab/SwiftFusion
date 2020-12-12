import Foundation
import PenguinStructures
import TensorFlow

/// A vector, in a Euclidean vector space with standard orthonormal basis.
//
// Note: Generalized vector spaces may not have Euclidean structure or a standard basis, so we may
// eventually add a "generalized vector" protocol. However, most vectors used in computations do
// have Euclidean structure and standard basis, so we'll reserve the short name `Vector` for
// such vectors and use a longer name like `GeneralizedVector` for the generalized vector spaces.
public protocol Vector: Differentiable where Self.TangentVector == Self {
  /// A type that can represent all of this vector's scalar values in a standard basis.
  associatedtype Scalars: MutableCollection where Scalars.Element == Double

  /// This vector's scalar values in a standard basis.
  var scalars: Scalars { get set }
  
  /// The number of components of this vector.
  var dimension: Int { get }

  // MARK: - AdditiveArithmetic requirements, refined to require differentiability.

  @differentiable
  static func += (_ lhs: inout Self, _ rhs: Self)

  @differentiable
  static func + (_ lhs: Self, _ rhs: Self) -> Self

  @differentiable
  static func -= (_ lhs: inout Self, _ rhs: Self)

  @differentiable
  static func - (_ lhs: Self, _ rhs: Self) -> Self

  // MARK: - Scalar multiplication.

  @differentiable
  static func *= (_ lhs: inout Self, _ rhs: Double)

  @differentiable
  static func * (_ lhs: Double, _ rhs: Self) -> Self

  /// A zero value of the same shape as `self`.
  var zeroValue: Self { get }

  // MARK: - Euclidean structure.

  /// The inner product of `self` with `other`.
  ///
  /// Note: Depends on Euclidean vector space structure, and therefore would not be available on a
  /// generalized vector.
  @differentiable
  func dot(_ other: Self) -> Double

  // MARK: - Methods for accessing the standard basis and for manipulating the coordinates under
  // the standard basis.

  #if true // set to false to find examples to be removed.
  // TODO(marcrasi): Remove these requirements!
  
  /// Returns the result of calling `body` on the scalars of `self`.
  ///
  /// Note: Depends on a determined standard basis, and therefore would not be available on a
  /// generalized vector.
  ///
  /// A default is provided that is correct for types that are represented as contiguous scalars
  /// in memory.
  func withUnsafeBufferPointer<R>(
    _ body: (UnsafeBufferPointer<Double>) throws -> R
  ) rethrows -> R

  /// Returns the result of calling `body` on the scalars of `self`.
  ///
  /// Note: Depends on a determined standard basis, and therefore would not be available on a
  /// generalized vector.
  ///
  /// A default is provided that is correct for types that are represented as contiguous scalars
  /// in memory.
  mutating func withUnsafeMutableBufferPointer<R>(
    _ body: (inout UnsafeMutableBufferPointer<Double>) throws -> R
  ) rethrows -> R
  #endif
}

extension Vector {
  /// A zero value of the same shape as `self`.
  public var zeroValue: Self { 0.0 * self }
}

/// A `Vector` whose instances can be initialized for a collection of scalars.
public protocol ScalarsInitializableVector: Vector {
  /// Creates an instance whose elements are `scalars`.
  ///
  /// Note: Depends on a determined standard basis, and therefore would not be available on a
  /// generalized vector.
  ///
  /// Precondition: `scalars` must have an element count that `Self` can hold (e.g. if `Self` is a
  /// fixed-size vectors, then `scalars` must have exactly the right number of elements).
  ///
  /// TODO: Maybe make this failable.
  init<Source: Collection>(_ scalars: Source) where Source.Element == Double
}

/// A collection of the standard basis values of `V` with a given shape.
///
/// The elements are the distinct vector values having a single non-zero scalar with value 1.
public struct StandardBasis<V: Vector>: Collection {
  /// A zero value of the same shape as the elements of `self`
  private let shapedZero: V

  /// Creates an instance containing the basis values with the same shape as `template`.
  ///
  /// - Requires: `template` is zero-valued.
  public init(shapedLikeZero template: V) {
    assert(template == template.zeroValue)
    shapedZero = template
  }

  /// Creates an instance containing the basis values with the same shape as `template`.
  public init(shapedLike template: V) {
    shapedZero = template.zeroValue
  }
  
  /// A position in the collection of basis vectors.
  public typealias Index = V.Scalars.Index
  
  /// The position of the first element, or `endIndex` if `self.isEmpty`.
  public var startIndex: Index { shapedZero.scalars.startIndex }
  
  /// The position one step beyond the last contained element.
  public var endIndex: Index { shapedZero.scalars.endIndex }
  
  /// Accesses the unit vector at `i`.
  public subscript(i: Index) -> V {
    var r = shapedZero
    r.scalars[i] = 1
    return r
  }

  /// Returns the position after `i`.
  ///
  /// - Requires: `i != endIndex`.
  public func index(after i: Index) -> Index {
    shapedZero.scalars.index(after: i)
  }

  /// Moves `i` to the next position.
  ///
  /// - Requires: `i != endIndex`.
  public func formIndex(after i: inout Index) {
    shapedZero.scalars.formIndex(after: &i)
  }
}
  
/// A `Vector` whose instances all have the same `dimension`.
///
/// Note: This is a temporary shim to help incrementally remove the assumption that vectors have a
/// static `dimension`. New code should prefer `Vector` so that it does not rely on a static `dimension`.
public protocol FixedSizeVector: ScalarsInitializableVector {
  /// The `dimension` of an instance.
  static var dimension: Int { get }
}

extension FixedSizeVector {
  /// A zero value of the same shape as `self`.
  public var zeroValue: Self {
    .init(repeatElement(0.0, count: dimension))
  }

  /// The standard basis values of `Self`.
  public static var standardBasis: StandardBasis<Self> {
    .init(shapedLikeZero: .init(repeatElement(0.0, count: dimension)))
  }
}

/// Vector space operations that can be implemented in terms of others.
extension Vector {
  @differentiable
  public static prefix func - (_ v: Self) -> Self {
    return (-1) * v
  }

  @differentiable
  public var squaredNorm: Double {
    return self.dot(self)
  }

  @differentiable
  public var norm: Double {
    return squaredNorm.squareRoot()
  }

  @differentiable
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    var result = lhs
    result += rhs
    return result
  }

  @differentiable
  public static func - (_ lhs: Self, _ rhs: Self) -> Self {
    var result = lhs
    result -= rhs
    return result
  }

  @differentiable
  public static func * (_ lhs: Double, _ rhs: Self) -> Self {
    var result = rhs
    result *= lhs
    return result
  }
}

#if true // set to false to find examples to be removed.
// TODO(marcrasi): Remove these implementations!

/// Default implementations of raw memory access.
extension Vector {
  /// Returns the result of calling `body` on the scalars of `self`.
  public func withUnsafeBufferPointer<R>(
    _ body: (UnsafeBufferPointer<Double>) throws -> R
  ) rethrows -> R {
    return try withUnsafePointer(to: self) { [dimension = self.dimension] p in
      try body(
          UnsafeBufferPointer<Double>(
              start: UnsafeRawPointer(p)
                  .assumingMemoryBound(to: Double.self),
              count: dimension))
    }
  }

  /// Returns the result of calling `body` on the scalars of `self`.
  public mutating func withUnsafeMutableBufferPointer<R>(
    _ body: (inout UnsafeMutableBufferPointer<Double>) throws -> R
  ) rethrows -> R {
    try withUnsafeMutablePointer(to: &self) { [dimension = self.dimension] p in
      var b = UnsafeMutableBufferPointer<Double>(
        start: UnsafeMutableRawPointer(p).assumingMemoryBound(to: Double.self),
        count: dimension)
      return try body(&b)
    }
  }
}
#endif

/// Conversion to `Tensor`.
extension Vector {
  /// Returns a `Tensor` with shape `[Self.dimension]` with the same scalars as `self`.
  @differentiable(where Self: ScalarsInitializableVector)
  public var flatTensor: Tensor<Double> {
    Tensor<Double>(shape: [scalars.count], scalars: Array(scalars))
  }

  @derivative(of: flatTensor)
  @usableFromInline
  func vjpFlatTensor() -> (value: Tensor<Double>, pullback: (Tensor<Double>) -> Self)
    where Self: ScalarsInitializableVector
  {
    return (self.flatTensor, { Self(flatTensor: $0) })
  }
}

/// Conversion from `Tensor`.
extension ScalarsInitializableVector {
  /// Creates an instance with the same scalars as `flatTensor`.
  @differentiable
  public init(flatTensor: Tensor<Double>) {
    self.init(flatTensor.scalars)
  }

  @derivative(of: init(flatTensor:))
  @usableFromInline
  static func vjpInit(flatTensor: Tensor<Double>) -> (
    value: Self,
    pullback: (Self) -> Tensor<Double>
  ) {
    return (Self(flatTensor: flatTensor), { $0.flatTensor })
  }
}

/// Conversions between `FixedSizeVector`s with compatible shapes.
extension FixedSizeVector {
  /// Creates a vector with the same scalars as `v`.
  ///
  /// - Requires: `Self.dimension == V.dimension`.
  @differentiable
  public init<V: FixedSizeVector>(_ v: V) {
    precondition(Self.dimension == V.dimension)
    self.init(v.scalars)
  }

  @derivative(of: init(_:))
  @usableFromInline
  static func vjpInit<V: FixedSizeVector>(_ v: V) -> (value: Self, pullback: (Self) -> V) {
    return (
      Self(v),
      { t in V(t) }
    )
  }

  /// Creates a vector with the scalars from `v1`, followed by the scalars from `v2`.
  ///
  /// - Requires: `Self.dimension == V1.dimension + V2.dimension`.
  @differentiable
  public init<V1: FixedSizeVector, V2: FixedSizeVector>(concatenating v1: V1, _ v2: V2) {
    precondition(Self.dimension == V1.dimension + V2.dimension)
    self.init(v1.scalars.concatenated(to: v2.scalars))
  }

  @derivative(of: init(concatenating:_:))
  @usableFromInline
  static func vjpInit<V1: FixedSizeVector, V2: FixedSizeVector>(concatenating v1: V1, _ v2: V2) -> (
    value: Self,
    pullback: (Self) -> (V1, V2)
  ) {
    return (
      Self(concatenating: v1, v2),
      { t in
        let p = t.scalars.index(atOffset: V1.dimension)
        return (.init(t.scalars[..<p]), .init(t.scalars[p...]))
      }
    )
  }
}
