/// A vector in a vector space.
public protocol VectorProtocol: AdditiveArithmetic {
  /// The scalar type of the vector space.
  ///
  /// Note that if the vector is a collection, this is not necessarily the same as the element
  /// type of the collection. For example, a matrix could be viewed as a collection of columns,
  /// but a matrix could simultaneously be viewed as a vector whose scalar is the element type of
  /// the columns.
  associatedtype Scalar: Numeric
  
  /// A vector in the dual space [1].
  ///
  /// [1] https://en.wikipedia.org/wiki/Dual_space
  associatedtype Covector: VectorProtocol
  where Covector.Scalar == Scalar, Covector.Covector == Self

  /// Adds `other` to `self` using vector space addition.
  ///
  /// TODO(TF-982): We can eliminate this requirement and use `AdditiveArithmetic.+=` instead.
  /// But we need default derivative implementations for protocol requirements (TF-982) first.
  mutating func add(_ other: Self)

  /// Scales `self` by `scalar` using vector space scalar multiplication.
  ///
  /// TODO(TF-982): We can eliminate this requirement and use `*=` instead.
  /// But we need default derivative implementations for protocol requirements (TF-982) first.
  mutating func scale(by scalar: Scalar)
  
  /// Returns the result of `covector` evaluated at `self`.
  ///
  /// In addition to "bracket", this is also known as the "natural pairing" or the "evaluation
  /// map" [1].
  ///
  /// Example: If `Self` is a Euclidean vector and we identify `Self == Covector`, then this is
  /// the inner product.
  ///
  /// [1] https://en.wikipedia.org/wiki/Dual_space
  func bracket(_ covector: Covector) -> Scalar
}

/// Default implementations of `+=`, `+`, `-=`, `-`, `*=`, and `*`.
extension VectorProtocol {
  public static func += (_ lhs: inout Self, _ rhs: Self) {
    lhs.add(rhs)
  }

  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    var result = lhs
    result += rhs
    return result
  }

  public static func -= (_ lhs: inout Self, _ rhs: Self) {
    let negRhs = (-1 as Scalar) * rhs
    lhs += negRhs
  }

  public static func - (_ lhs: Self, _ rhs: Self) -> Self {
    var result = lhs
    result -= rhs
    return result
  }

  public static func *= (_ lhs: inout Self, _ rhs: Scalar) {
    lhs.scale(by: rhs)
  }

  public static func * (_ lhs: Scalar, _ rhs: Self) -> Self {
    var result = rhs
    result *= lhs
    return result
  }
}
