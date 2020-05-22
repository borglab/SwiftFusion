/// A vector in a Euclidean vector space whose dimension is fixed at compile time.
public protocol EuclideanVectorN: EuclideanVector {
  /// The dimension of the vector.
  static var dimension: Int { get }

  /// A standard basis of vectors.
  static var standardBasis: [Self] { get }
}
