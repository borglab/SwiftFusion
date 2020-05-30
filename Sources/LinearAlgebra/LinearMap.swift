/// A linear map from `Input` to `Output`.
public protocol LinearMap {
  /// The input type.
  associatedtype Input: VectorProtocol

  /// The output type.
  associatedtype Output: VectorProtocol

  /// Returns the result of the linear map applied to `x`.
  func forward(_ x: Input) -> Output
}
