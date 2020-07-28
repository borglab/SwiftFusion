import Foundation
import TensorFlow

/// An element of a Euclidean vector space.
public protocol EuclideanVector: Differentiable where Self.TangentVector == Self {
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
  @differentiable
  func dot(_ other: Self) -> Double

  // MARK: - Methods for setting/accessing scalars.

  /// Creates an instance whose elements are `scalars`.
  ///
  /// Precondition: `scalars` must have an element count that `Self` can hold (e.g. if `Self` is a
  /// fixed-size vectors, then `scalars` must have exactly the right number of elements).
  ///
  /// TODO: Maybe make this failable.
  init<Source: Collection>(_ scalars: Source) where Source.Element == Double

  /// Returns the result of calling `body` on the scalars of `self`.
  ///
  /// A default is provided that returns a pointer to `self`.
  func withUnsafeBufferPointer<R>(
    _ body: (UnsafeBufferPointer<Double>) throws -> R
  ) rethrows -> R

  /// Returns the result of calling `body` on the scalars of `self`.
  ///
  /// A default is provided that returns a pointer to `self`.
  mutating func withUnsafeMutableBufferPointer<R>(
    _ body: (UnsafeMutableBufferPointer<Double>) throws -> R
  ) rethrows -> R

  // MARK: - Methods relating to static dimension.
  // TODO: Make these per-instance so that we can support dynamically-sized vectors.

  /// The dimension of the vector.
  static var dimension: Int { get }

  /// A standard basis of vectors.
  static var standardBasis: [Self] { get }
}

/// Convenient operations on Euclidean vector spaces that can be implemented in terms of the
/// primitive operations.
extension EuclideanVector {
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
}

/// Default implementations of some vector space operations in terms of other vector space
/// operations.
extension EuclideanVector {
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

/// Default implementations of raw memory access, for vectors represented as contiguous scalars.
extension EuclideanVector {
  /// Returns the result of calling `body` on the scalars of `self`.
  public func withUnsafeBufferPointer<R>(
    _ body: (UnsafeBufferPointer<Double>) throws -> R
  ) rethrows -> R {
    return try withUnsafePointer(to: self) { p in
      try body(
          UnsafeBufferPointer<Double>(
              start: UnsafeRawPointer(p)
                  .assumingMemoryBound(to: Double.self),
              count: Self.dimension))
    }
  }

  /// Returns the result of calling `body` on the scalars of `self`.
  public mutating func withUnsafeMutableBufferPointer<R>(
    _ body: (UnsafeMutableBufferPointer<Double>) throws -> R
  ) rethrows -> R {
    return try withUnsafeMutablePointer(to: &self) { p in
      try body(
          UnsafeMutableBufferPointer<Double>(
              start: UnsafeMutableRawPointer(p)
                  .assumingMemoryBound(to: Double.self),
              count: Self.dimension))
    }
  }
}

/// Implementation of `subscript`.
extension EuclideanVector {
  /// Accesses the scalar at `i`.
  subscript(i: Int) -> Double {
    _read {
      boundsCheck(i)
      yield withUnsafeBufferPointer { $0.baseAddress.unsafelyUnwrapped[i] }
    }
    _modify {
      boundsCheck(i)
      defer { _fixLifetime(self) }
      yield &withUnsafeMutableBufferPointer { $0.baseAddress }.unsafelyUnwrapped[i]
    }
  }

  /// Traps with a suitable error message if `i` is not the position of an
  /// element in `self`.
  private func boundsCheck(_ i: Int) {
    precondition(i >= 0 && i < Self.dimension, "index out of range")
  }
}

/// Conversions between `Vector`s with compatible shapes.
extension EuclideanVector {
  /// Creates a vector with the same scalars as `v`.
  ///
  /// - Requires: `Self.dimension == V.dimension`.
  @differentiable
  public init<V: EuclideanVector>(_ v: V) {
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
  static func vjpInit<V: EuclideanVector>(_ v: V) -> (value: Self, pullback: (Self) -> V) {
    return (
      Self(v),
      { t in V(t) }
    )
  }

  /// Creates a vector with the scalars from `v1`, followed by the scalars from `v2`.
  ///
  /// - Requires: `Self.dimension == V1.dimension + V2.dimension`.
  @differentiable
  public init<V1: EuclideanVector, V2: EuclideanVector>(concatenating v1: V1, _ v2: V2) {
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
  static func vjpInit<V1: EuclideanVector, V2: EuclideanVector>(concatenating v1: V1, _ v2: V2) -> (
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

/// Conversions to/from `Tensor`.
extension EuclideanVector {
  /// Creates an instance with the same scalars as `flatTensor`.
  ///
  /// - Reqiures: `flatTensor.shape == [Self.dimension]`.
  @differentiable
  public init(flatTensor: Tensor<Double>) {
    precondition(flatTensor.shape == [Self.dimension])
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

  /// Returns a `Tensor` with shape `[Self.dimension]` with the same scalars as `self`.
  @differentiable
  public var flatTensor: Tensor<Double> {
    withUnsafeBufferPointer { b in
      return Tensor<Double>(shape: [b.count], scalars: b)
    }
  }

  @derivative(of: flatTensor)
  @usableFromInline
  func vjpFlatTensor() -> (value: Tensor<Double>, pullback: (Tensor<Double>) -> Self) {
    return (self.flatTensor, { Self(flatTensor: $0) })
  }
}
