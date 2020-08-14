import Foundation
import TensorFlow

/// A vector, in a Euclidean vector space with standard orthonormal basis.
//
// Note: Generalized vector spaces may not have Euclidean structure or a standard basis, so we may
// eventually add a "generalized vector" protocol. However, most vectors used in computations do
// have Euclidean structure and standard basis, so we'll reserve the short name `Vector` for
// such vectors and use a longer name like `GeneralizedVector` for the generalized vector spaces.
public protocol Vector: Differentiable where Self.TangentVector == Self {
  associatedtype Scalars: MutableCollection where Scalars.Element == Double

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

  // MARK: - Euclidean structure.

  /// The inner product of `self` with `other`.
  ///
  /// Note: Depends on Euclidean vector space structure, and therefore would not be available on a
  /// generalized vector.
  @differentiable
  func dot(_ other: Self) -> Double
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

/// A `Vector` whose instances all have the same `dimension`.
///
/// Note: This is a temporary shim to help incrementally remove the assumption that vectors have a
/// static `dimension`. New code should prefer `Vector` so that it does not rely on a static `dimension`.
public protocol FixedSizeVector: ScalarsInitializableVector {
  /// The `dimension` of an instance.
  static var dimension: Int { get }
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
    _ body: (UnsafeMutableBufferPointer<Double>) throws -> R
  ) rethrows -> R {
    return try withUnsafeMutablePointer(to: &self) { [dimension = self.dimension] p in
      try body(
          UnsafeMutableBufferPointer<Double>(
              start: UnsafeMutableRawPointer(p)
                  .assumingMemoryBound(to: Double.self),
              count: dimension))
    }
  }
}

/// Conversion to `Tensor`.
extension Vector {
  /// Returns a `Tensor` with shape `[Self.dimension]` with the same scalars as `self`.
  @differentiable(where Self: ScalarsInitializableVector)
  public var flatTensor: Tensor<Double> {
    withUnsafeBufferPointer { b in
      return Tensor<Double>(shape: [b.count], scalars: b)
    }
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
    self = Self.zero
    self.withUnsafeMutableBufferPointer { rBuf in
      v.withUnsafeBufferPointer { vBuf in
        for (i, s) in vBuf.enumerated() {
          rBuf[i] = s
        }
      }
    }
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
    self = Self.zero
    self.withUnsafeMutableBufferPointer { rBuf in
      v1.withUnsafeBufferPointer { v1Buf in
        for (i, s) in v1Buf.enumerated() {
          rBuf[i] = s
        }
      }
      v2.withUnsafeBufferPointer { v2Buf in
        for (i, s) in v2Buf.enumerated() {
          rBuf[i + V1.dimension] = s
        }
      }
    }
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
        t.withUnsafeBufferPointer { tBuf in
          var t1 = V1.zero
          t1.withUnsafeMutableBufferPointer { t1Buf in
            for i in t1Buf.indices {
              t1Buf[i] = tBuf[i]
            }
          }
          var t2 = V2.zero
          t2.withUnsafeMutableBufferPointer { t2Buf in
            for i in t2Buf.indices {
              t2Buf[i] = tBuf[i + V1.dimension]
            }
          }
          return (t1, t2)
        }
      }
    )
  }
}
