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

/// This file tests estimating a mixture of Gaissian with Monte Carle EM

import SwiftFusion
import TensorFlow
import XCTest

// Simple two-component mixture model in 2D
struct TwoComponents : McEmModel {
  typealias Datum = Tensor<Double>
  enum Hidden { case one; case two}
  
  var c1, c2 : MultivariateGaussian
  
  /// Initialize to uninitialized components
  init(_ data:[Datum], using sourceOfEntropy: inout AnyRandomNumberGenerator) {
    c1 = MultivariateGaussian(dims:[2], regularizer:1e-2)
    c1.covariance_inv = eye(rowCount: 2)
    c1.mean = data[0]
    c2 = MultivariateGaussian(dims:[2], regularizer:1e-2)
    c2.covariance_inv = eye(rowCount: 2)
    c2.mean = data[1]
  }
  
  /// Given a datum and a model, sample from the hidden variables
  func sample(count:Int, for datum: Datum,
              using sourceOfEntropy: inout AnyRandomNumberGenerator) -> [Hidden] {
    let p1 = c1.probability(datum)
    let p2 = c2.probability(datum)
    let labels : [Hidden] = (0..<count).map { _ in
      let u = Double.random(in: 0..<p1+p2, using: &sourceOfEntropy)
      return u<=p1 ? .one : .two
    }
    return labels
  }
  
  /// Given an array of labeled datums, fit the two Gaussian mixture components
  mutating func fit(_ labeledData: [LabeledDatum]) {
    let data1 = labeledData.filter { $0.0 == .one}
    let data2 = labeledData.filter { $0.0 == .two}
    c1.fit(Tensor<Double>(data1.map { $0.1 }))
    c2.fit(Tensor<Double>(data2.map { $0.1 }))
  }
}

final class MonteCarloEMTests: XCTestCase {
  let data = [[1.0, 2.0], [1.0, 2.1], [6.0, 8.0], [6.2, 7.9]].map { Tensor<Double>($0)}
  
  let deterministicEntropy = ARC4RandomNumberGenerator(seed: 42)
  
  /// Test fitting a simple 2-component mixture
  func testTwoComponents() {
    var em = MonteCarloEM<TwoComponents>(sourceOfEntropy: deterministicEntropy)
    let model : TwoComponents = em.run(with:data, iterationCount: 5)
    assertEqual(model.c1.mean!, Tensor<Double>([1.0, 2.0]), accuracy: 0.2)
    assertEqual(model.c2.mean!, Tensor<Double>([6.0, 8.0]), accuracy: 0.2)
  }
}
