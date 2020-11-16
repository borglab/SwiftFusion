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
  public typealias T = Tensor<Double>
  public typealias HyperParameters = Double /// Just the regularizer
  
  public let mean: T /// mean
  public let information: T /// Information matrix
  public let constant : Double /// normalization constant
  
  /**
   Initialize a Multivariate Gaussian Model
   - Parameters:
    - mean: n-dimensional vector
    - information: information matrix of shape [n,n]
   */
  public init(mean: T, information: T) {
    precondition(mean.rank == 1, "mean has to be a vector")
    let n = mean.shape[0]
    precondition(information.shape == [n,n], "information has to be nxn")
    self.mean = mean
    self.information = information
    self.constant = sqrt(_Raw.matrixDeterminant(information/(2.0 * .pi)).scalarized())
  }
  
  /// Initalize by fitting the model to the data
  ///  - data: Tensor of shape [N, <dims>]
  public init(from data: T, given p:HyperParameters? = nil) {
    assert(data.shape.dropFirst().rank == 1)
    let mean = data.mean(squeezingAxes: 0)
    let cov_data = cov(data)
    let r = p ?? 1e-10
    let regularized_cov = cov_data.withDiagonal(cov_data.diagonalPart() + r)
    self.init(mean:mean, information: pinv(regularized_cov))
  }
  
  /// Calculated the negative log likelihood of *one* data point
  /// Note this is NOT normalized probability
  @differentiable public func negativeLogLikelihood(_ sample: T) -> Double {
    precondition(sample.shape == mean.shape)
    let normalized = (sample - mean).expandingShape(at: 1)
    /// FIXME: this is a workaround for bug in the derivative of `.scalarized()`
    let t = matmul(normalized, transposed: true, matmul(information, normalized)).withDerivative {
      $0 = $0.reshaped(to: [1, 1])
    }
    
    return t.scalarized() / 2.0
  }
  
  /// Calculated normalized probability
  @differentiable public func probability(_ sample: T) -> Double {
    // - ToDo: Precalculate constant
    let E = negativeLogLikelihood(sample)
    return exp(-E) * self.constant
  }
}
