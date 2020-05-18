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
}
