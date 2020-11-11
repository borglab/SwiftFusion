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

    /// Compare with the NumPy implementation
    func testSimpleCovMatrix() {
        let data = Tensor<Double>([[1.5, 0.9], [0.5, 1.9]])

        assertEqual(cov(data), Tensor<Double>([[0.5, -0.5], [-0.5, 0.5]]), accuracy: 1e-8)

        let data_zerovar = Tensor<Double>([[1.0, 2.0, 3.0], [1.0, 2.0, 3.1], [0.9, 2.0, 2.9]])
        assertEqual(cov(data_zerovar), Tensor<Double>(
            [[0.00333333, 0       , 0.005     ],
            [0       , 0        , 0        ],
            [0.005     , 0        , 0.01      ]]), accuracy: 1e-8)

    }
}