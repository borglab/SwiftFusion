/// A vector in a Euclidean vector space whose dimension is fixed at compile time.
public protocol EuclideanVectorN: EuclideanVector {
  /// The dimension of the vector.
  static var dimension: Int { get }

  /// A standard basis of vectors.
  static var standardBasis: [Self] { get }
}

extension EuclideanVectorN {
  /// Returns the result of calling `body` on the scalars of `self`.
  func withUnsafeBufferPointer<R>(
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
  mutating func withUnsafeMutableBufferPointer<R>(
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
