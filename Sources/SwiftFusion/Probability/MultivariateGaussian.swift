// Copyright 2019 The SwiftFusion Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import TensorFlow

/// Calculate the covariance of a data matrix
/// preconditon: X.rank == 2
public func cov(_ X: Tensor<Double>) -> Tensor<Double> {
    precondition(X.rank == 2, "cov() can only handle 2D data")
    let X_norm = X - X.mean(squeezingAxes: 0)
    let N = X.shape[0]
    return matmul(X_norm.transposed(), X_norm) / Double(N - 1)
}

/// A Multivariate Gaussian Density
///
/// This is a density where all dimensions
/// share one multivariate Gaussian density.
public struct MultivariateGaussian: GenerativeDensity {
  public let dims: TensorShape

  /// Sample standard deviation
  public var covariance_inv: Optional<Tensor<Double>> = nil

  /// Sample Mean
  public var mean: Optional<Tensor<Double>> = nil

  /// This avoids division by zero when the data is zero variance
  public var regularizer: Double

  /// Initialize a Multivariate Gaussian Model
  /// Note: only supports 1-d data points
  public init(dims: TensorShape, regularizer: Double = 1e-10) {
    precondition(dims.count == 1, "Multivariate Gaussian only support 1-d tensor input")
    self.dims = dims
    self.regularizer = regularizer
  }

  /// Fit the model to the data
  ///  - data: Tensor of shape [N, <dims>]
  public mutating func fit(_ data: Tensor<Double>) {
    assert(data.shape.dropFirst() == dims)

    self.mean = data.mean(squeezingAxes: 0)
    let cov_data = cov(data)
    print("Cov: \(cov_data)")
    let regularized_cov = cov_data.withDiagonal(cov_data.diagonalPart() + regularizer)
    print("CovR: \(regularized_cov)")
    self.covariance_inv = pinv(regularized_cov)
  }

  /// Calculated the negative log likelihood of *one* data point
  /// Note this is NOT normalized probability
  @differentiable public func negativeLogLikelihood(_ sample: Tensor<Double>) -> Double {
    precondition(sample.shape == self.dims)

    let normalized = (sample - mean!).expandingShape(at: 1)
    let t = matmul(normalized.transposed(), matmul(covariance_inv!, normalized))

    return t.scalarized()
  }
}