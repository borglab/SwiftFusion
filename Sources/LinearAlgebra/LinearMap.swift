/// A linear map from `Input` to `Output`.
public protocol LinearMap {
  /// The input type.
  associatedtype Input: VectorProtocol

  /// The output type.
  associatedtype Output: VectorProtocol

  /// Returns the result of the linear map applied to `x`.
  func apply(_ x: Input) -> Output
}
