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

// import TensorFlow
import _Differentiation

/// A Gaussian Naive Bayes density
///
/// This is a density where each dimension has its own 1-d Gaussian density.
public struct GaussianNB: GenerativeDensity {
  public typealias T = Tensor<Double>
  public typealias HyperParameters = Double /// Just the regularizer
  
  public let mu: T  /// Sample Mean
  public let sigmas: T /// Sample standard deviation
  public let precisions: T /// Cached precisions
  
  /** Initalize by fitting the model to the data
   - Parameters:
   - data: Tensor of shape [N, <dims>]
   - regularizer: avoids division by zero when the data is zero variance
   */
  public init(from data: T, regularizer r: Double) {
    self.mu = data.mean(squeezingAxes: 0)
    let sigmas = data.standardDeviation(squeezingAxes: 0) + r
    self.sigmas = sigmas
    self.precisions = 1.0 / sigmas.squared()
  }
  
  /// Initalize by fitting the model to the data
  ///  - data: Tensor of shape [N, <dims>]
  public init(from data: T, given p:HyperParameters? = nil) {
    self.init(from:data, regularizer: p ?? 1e-10)
  }
  
  /// Calculated the negative log likelihood of *one* data point
  /// Note this is NOT normalized probability
  @differentiable(reverse) public func negativeLogLikelihood(_ sample: T) -> Double {
    precondition(sample.shape == mu.shape)
    let t = (sample - mu).squared() * precisions
    return t.sum().scalarized() / 2.0
  }
}
