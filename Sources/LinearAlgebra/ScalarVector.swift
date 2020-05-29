/// Views `Scalar` as a vector.
public struct ScalarVector<Scalar> {
  var scalar: Scalar
  
  public init(_ scalar: Scalar) {
    self.scalar = scalar
  }
}

extension ScalarVector: Equatable where Scalar: Equatable {}

extension ScalarVector: ExpressibleByFloatLiteral where Scalar: ExpressibleByFloatLiteral {
  public init(floatLiteral value: Scalar.FloatLiteralType) {
    self.scalar = Scalar(floatLiteral: value)
  }
}

extension ScalarVector: ExpressibleByIntegerLiteral where Scalar: ExpressibleByIntegerLiteral {
  public init(integerLiteral value: Scalar.IntegerLiteralType) {
    self.scalar = Scalar(integerLiteral: value)
  }
}

extension ScalarVector: AdditiveArithmetic, Vector where Scalar: Numeric {
  public typealias Covector = Self
  public mutating func add(_ other: Self) {
    self.scalar += other.scalar
  }
  public mutating func scale(by scalar: Scalar) {
    self.scalar *= scalar
  }
  public func bracket(_ v: Self) -> Scalar {
    return self.scalar * v.scalar
  }
  public static var zero: Self { return Self(0) }
}

extension ScalarVector: Differentiable, DifferentiableVector
where Scalar: Differentiable, Scalar: Numeric, Scalar.TangentVector == Scalar {
  public typealias TangentVector = Covector
}
