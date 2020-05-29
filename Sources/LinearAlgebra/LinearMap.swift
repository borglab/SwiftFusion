/// A linear map from `Input` to `Output`.
public protocol LinearMap {
  /// The input type.
  associatedtype Input: Vector

  /// The output type.
  associatedtype Output: Vector

  /// Returns the result of the linear map applied to `x`.
  func callAsFunction(_ x: Input) -> Output
}

/// A `Vector` is a linear map from `ScalarVector<Scalar>` to `Vector`!
extension Vector {
  public typealias Input = ScalarVector<Scalar>
  public typealias Output = Self
  public func callAsFunction(_ x: Input) -> Output {
    return x.scalar * self
  }
}
