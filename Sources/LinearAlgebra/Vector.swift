/// A vector in a vector space.
public protocol Vector: AdditiveArithmetic {
  /// The scalar type of the vector space.
  ///
  /// Note that if the vector is a collection, this is not necessarily the same as the element
  /// type of the collection. For example, a matrix could be viewed as a collection of columns,
  /// but a matrix could simultaneously be viewed as a vector whose scalar is the element type of
  /// the columns.
  associatedtype Scalar: Numeric

  /// Adds `other` to `self` using vector space addition.
  ///
  /// TODO(TF-982): We can eliminate this requirement and use `AdditiveArithmetic.+=` instead.
  /// But we need default derivative implementations for protocol requirements (TF-982) first.
  mutating func add(_ other: Self)

  /// Scales `self` by `scalar` using vector space scalar multiplication.
  mutating func scale(by scalar: Scalar)
}

/// Default implementations of `+=`, `+`, `-=`, `-`, `*=`, and `*`.
extension Vector {
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
