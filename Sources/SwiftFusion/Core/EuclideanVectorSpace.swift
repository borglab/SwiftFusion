/// A Euclidean vector space.
public protocol EuclideanVectorSpace: Differentiable, VectorProtocol
  where Self.TangentVector == Self
{
  // Note: This is a work in progress. We intend to add more requirements here as we need them.
}
