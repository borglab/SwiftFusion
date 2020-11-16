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

/// This file tests proper Multivariate Gaussian behavior

import SwiftFusion
import TensorFlow
import XCTest

final class MultivariateGaussianTests: XCTestCase {
  typealias T = Tensor<Double>
  
  /// Compare with the NumPy implementation
  func testSimpleCovMatrix() {
    let data = T([[1.5, 0.9], [0.5, 1.9]])
    
    assertEqual(cov(data), T([[0.5, -0.5], [-0.5, 0.5]]), accuracy: 1e-8)
    
    let data_zerovar = T([[1.0, 2.0, 3.0], [1.0, 2.0, 3.1], [0.9, 2.0, 2.9]])
    assertEqual(cov(data_zerovar), T(
                  [[0.00333333, 0       , 0.005     ],
                   [0       , 0        , 0        ],
                   [0.005     , 0        , 0.01      ]]), accuracy: 1e-8
    )
  }
  
  /// Test  negative log likelihood
  func testMultivariateNegativeLogLikelihood() {
    let data = T([[1.5, 0.9], [0.5, 1.9], [1.0, 1.5]])
    
    let model = MultivariateGaussian(from: data)
        
    XCTAssertEqual(
      model.negativeLogLikelihood(T([1.0, 1.5])),
      1.33333333 / 2.0,
      accuracy: 1e-5
    )
  }
  
  /// Test  negative log likelihood and probability for simple case
  func testMultivariateProbability() {
    var model = MultivariateGaussian(dims:[2])
    model.covariance_inv = eye(rowCount: 2)
    model.mean = T(zeros:[2])
    
    let v = T([1, 2])
    let expectedE : Double = (1+4)/2
    XCTAssertEqual(model.negativeLogLikelihood(v), expectedE, accuracy: 1e-5)
    XCTAssertEqual(model.probability(v), exp(-expectedE)/sqrt(4 * .pi * .pi), accuracy: 1e-5)
  }
}
