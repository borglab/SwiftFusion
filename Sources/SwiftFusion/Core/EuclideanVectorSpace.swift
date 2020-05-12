import Foundation

/// A Euclidean vector space.
public protocol EuclideanVectorSpace: Differentiable, VectorProtocol
  where Self.TangentVector == Self, Self.VectorSpaceScalar == Double
{
  // Note: This is a work in progress. We intend to add more requirements here as we need them.

  /// The squared Euclidean norm of `self`.
  var squaredNorm: Double { get }
}

/// Convenient operators on Euclidean vector spaces.
extension EuclideanVectorSpace {
  /// The Euclidean norm of `self`.
  public var norm: Double {
    return sqrt(squaredNorm)
  }

  // Note: We can't have these because Swift type inference is very inefficient
  // and these make it too slow.
  //
  // public static func * (_ lhs: Double, _ rhs: Self) -> Self {
  //   return lhs.scaled(by: lhs)
  // }
  //
  // public static func * (_ lhs: Self, _ rhs: Double) -> Self {
  //   return lhs.scaled(by: rhs)
  // }
  //
  // public static func / (_ lhs: Self, _ rhs: Double) -> Self {
  //   return lhs.scaled(by: 1 / rhs)
  // }
}
