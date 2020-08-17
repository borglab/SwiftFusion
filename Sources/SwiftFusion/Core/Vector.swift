import Foundation
import TensorFlow

/// A vector, in a Euclidean vector space with standard orthonormal basis.
//
// Note: Generalized vector spaces may not have Euclidean structure or a standard basis, so we may
// eventually add a "generalized vector" protocol. However, most vectors used in computations do
// have Euclidean structure and standard basis, so we'll reserve the short name `Vector` for
// such vectors and use a longer name like `GeneralizedVector` for the generalized vector spaces.
public protocol Vector: Differentiable where Self.TangentVector == Self {
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

  // MARK: - Methods for accessing the standard basis and for manipulating the coordinates under
  // the standard basis.

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
    _ body: (UnsafeMutableBufferPointer<Double>) throws -> R
  ) rethrows -> R
}

/// A `Vector` whose instances all have the same `dimension`.
///
/// Note: This is a temporary shim to help incrementally remove the assumption that vectors have a
/// static `dimension`. New code should prefer `Vector` so that it does not rely on a static `dimension`.
public protocol FixedSizeVector: Vector {
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
