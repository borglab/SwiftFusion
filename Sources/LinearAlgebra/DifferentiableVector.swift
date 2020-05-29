/// A vector that can be used with differentiation.
///
/// Note that conformances do not need to implement any of the requirements because default
/// implementations with derivatives are provided.
///
/// We require `Self == Covector == TangentVector` because Swift AD APIs do not
/// currently support distinguishing between tangent vectors and cotangent vectors.
public protocol DifferentiableVector: Differentiable, VectorProtocol
where Scalar: Differentiable, Scalar.TangentVector == Scalar,
      Covector == Self,
      TangentVector == Covector
{
  /// Differentiable refinement of `AdditiveArithmetic.+=`.
  @differentiable
  static func += (_ lhs: inout Self, _ rhs: Self)

  /// Differentiable refinement of `AdditiveArithmetic.+`.
  @differentiable
  static func + (_ lhs: Self, _ rhs: Self) -> Self

  /// Differentiable refinement of `AdditiveArithmetic.-=`.
  @differentiable
  static func -= (_ lhs: inout Self, _ rhs: Self)

  /// Differentiable refinement of `AdditiveArithmetic.-`.
  @differentiable
  static func - (_ lhs: Self, _ rhs: Self) -> Self
}

/// A _differentiable_ bracket method.
///
/// TODO(TF-982): We can eliminate this and refine `Vector.bracket` to be differentiable.
/// But we need default derivative implementations for protocol requirements (TF-982) first.
extension DifferentiableVector {
  @differentiable
  public func differentiableBracket(_ covector: Covector) -> Scalar {
    return self.bracket(covector)
  }
  
  @derivative(of: differentiableBracket)
  @usableFromInline
  func vjpDifferentiableBracket(_ covector: Covector) ->
    (value: Scalar, pullback: (Scalar) -> (TangentVector, TangentVector))
  {
    return (
      self.differentiableBracket(covector),
      { (scalarTangent: Scalar) -> (TangentVector, TangentVector) in
        return (scalarTangent * covector, scalarTangent * self)
      }
    )
  }
}

/// Default implementations of _differentiable_ `+=`, `+`, `-=`, `-`, `*=`, and `*`.
extension DifferentiableVector {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
    plusEqualsImpl(&lhs, rhs)
  }

  @differentiable
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    var result = lhs
    result += rhs
    return result
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
    let negRhs = (-1 as Scalar) * rhs
    lhs += negRhs
  }

  @differentiable
  public static func - (_ lhs: Self, _ rhs: Self) -> Self {
    var result = lhs
    result -= rhs
    return result
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Scalar) {
    timesEqualsImpl(&lhs, rhs)
  }

  @differentiable
  public static func * (_ lhs: Scalar, _ rhs: Self) -> Self {
    var result = rhs
    result *= lhs
    return result
  }
  
  @differentiable
  private static func plusEqualsImpl(_ lhs: inout Self, _ rhs: Self) {
    lhs.add(rhs)
  }
  
  @differentiable
  private static func timesEqualsImpl (_ lhs: inout Self, _ rhs: Scalar) {
    lhs.scale(by: rhs)
  }
  
  @derivative(of: plusEqualsImpl)
  private static func vjpPlusEquals(_ lhs: inout Self, _ rhs: Self) -> (
    value: (), pullback: (inout TangentVector) -> TangentVector
  ) {
    lhs += rhs
    return ((), { $0 })
  }
  
  @derivative(of: timesEqualsImpl)
  private static func vjpTimesEquals(_ lhs: inout Self, _ rhs: Scalar) -> (
    value: (), pullback: (inout TangentVector) -> Scalar
  ) {
    let originalLhs = lhs
    lhs *= rhs
    return (
      (),
      { (tangent: inout TangentVector) -> Scalar in
        let rhsTangent = originalLhs.bracket(tangent)
        tangent.scale(by: rhs)
        return rhsTangent
      }
    )
  }
}
