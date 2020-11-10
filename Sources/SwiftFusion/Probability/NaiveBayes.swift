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


public struct GaussianNB: GaussianModel {

    public let dims: TensorShape

    public var sigmas: Optional<Tensor<Double>> = nil

    public var mus: Optional<Tensor<Double>> = nil

    /// Initialize a Gaussian Naive Bayes error model
    public init(dims: TensorShape) {
        self.dims = dims
    }

    public mutating func fit(_ data: Tensor<Double>) {
       assert(data.shape.dropFirst() == dims)

    }

    @differentiable public func negativeLogLikelihood(_ data: Tensor<Double>) -> Double {

        return 0
    }
}