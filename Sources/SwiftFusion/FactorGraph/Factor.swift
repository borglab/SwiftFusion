import TensorFlow

/// Computes measurement errors.
public protocol Factor {
  associatedtype Input: TensorConvertible where Input.TangentVector: TensorConvertible
  associatedtype Output: TensorConvertible where Output.TangentVector: TensorConvertible

  /// Returns the `error` of the factor.
  @differentiable
  func error(_ input: Input) -> Output
}

/// A `Factor` whose input and output types are `Tensor`s.
public protocol TensorFactor {
  /// Returns the `error` of the factor.
  @differentiable
  func error(tensorValues: Tensor<Double>) -> Tensor<Double>
}

extension Factor {
  @differentiable
  public func error(tensorValues: Tensor<Double>) -> Tensor<Double> {
    return error(Input(tensorValues)).tensor
  }

  // TODO(TF-1234): This custom derivative should not be necessary.
  @derivative(of: error(tensorValues:))
  @usableFromInline
  func errorWithPullback(tensorValues: Tensor<Double>) -> (
    value: Tensor<Double>,
    pullback: (Tensor<Double>) -> Tensor<Double>
  ) {
    let (err, pb) = valueWithPullback(at: Input(tensorValues), in: error)
    return (value: err.tensor, pullback: { pb(Output.TangentVector($0)).tensor })
  }
}
