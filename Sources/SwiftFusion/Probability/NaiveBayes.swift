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

/// A Gaussian Naive Bayes density
///
/// This is a density where each dimension has its own 1-d Gaussian density.
public struct GaussianNB: GenerativeDensity {
  public let dims: TensorShape

  /// Sample standard deviation
  public var sigmas: Optional<Tensor<Double>> = nil

  /// Sample Mean
  public var mus: Optional<Tensor<Double>> = nil

  /// Cached variance
  public var sigma2s: Optional<Tensor<Double>> = nil

  /// This avoids division by zero when the data is zero variance
  public var regularizer: Double

  /// Initialize a Gaussian Naive Bayes error model
  public init(dims: TensorShape, regularizer: Double = 1e-10) {
    self.dims = dims
    self.regularizer = regularizer
  }

  /// Initalize by fitting the model to the data
  ///  - data: Tensor of shape [N, <dims>]
  public typealias HyperParameters = ()
  public init(from data: Tensor<Double>, given p:HyperParameters? = nil) {
    self.init(dims: data.shape.dropFirst())
    fit(data)
  }

  /// Fit the model to the data
  ///  - data: Tensor of shape [N, <dims>]
  public mutating func fit(_ data: Tensor<Double>) {
    assert(data.shape.dropFirst() == dims)

    mus = data.mean(squeezingAxes: 0)
    sigmas = data.standardDeviation(squeezingAxes: 0) + regularizer
    sigma2s = sigmas!.squared()
  }

  /// Calculated the negative log likelihood of *one* data point
  /// Note this is NOT normalized probability
  @differentiable public func negativeLogLikelihood(_ sample: Tensor<Double>) -> Double {
    precondition(sample.shape == self.dims)

    let t = (sample - mus!).squared() / (2.0 * sigma2s!)

    return t.sum().scalarized()
  }
}
