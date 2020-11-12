import TensorFlow

/// The mean and standard deviation of a collection of frames.
public struct FrameStatistics {
  public var mean: Tensor<Double>
  public var standardDeviation: Tensor<Double>

  /// Creates an instance containing the statistics for `frames`.
  public init(_ frames: Tensor<Double>) {
    self.mean = frames.mean()
    self.standardDeviation = frames.standardDeviation()
  }

  /// Returns `v`, normalized to have mean `0` and standard deviation `1`.
  public func normalized(_ v: Tensor<Double>) -> Tensor<Double> {
    return (v - mean) / standardDeviation
  }

  /// Returns `n` scaled and shifted so that its mean and standard deviation are `self.mean`
  /// and `self.standardDeviation`.
  ///
  /// Requires that `n` has mean `0` and standard deviation `1`.
  public func unnormalized(_ n: Tensor<Double>) -> Tensor<Double> {
    return n * standardDeviation + mean
  }
}
