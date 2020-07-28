import Foundation
import TensorFlow

/// A vector in a Euclidean vector space.
public protocol EuclideanVector: Differentiable where Self.TangentVector == Self {
  // Note: This is a work in progress. We intend to add more requirements here as we need them.

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

/// Default implementations of some `EuclideanVector` requirements.
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
