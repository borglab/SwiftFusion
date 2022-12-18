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

/// This file tests estimating a mixture of 2 Gaussians with Monte Carlo EM

import SwiftFusion
// import TensorFlow
import XCTest

// Simple two-component mixture model in 2D
struct TwoComponents : McEmModel {
  typealias Datum = Tensor<Double>
  enum Hidden { case one; case two }
  typealias HyperParameters = MultivariateGaussian.HyperParameters
  
  var c1, c2 : MultivariateGaussian
  
  /// Initialize to uninitialized components
  init(from data:[Datum],
       using sourceOfEntropy: inout AnyRandomNumberGenerator,
       given p: HyperParameters?) {
    c1 = MultivariateGaussian(mean:data[0], information: eye(rowCount: 2))
    c2 = MultivariateGaussian(mean:data[1], information: eye(rowCount: 2))
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
  init(from labeledData: [LabeledDatum], given p: HyperParameters?=nil) {
    let data1 = labeledData.filter { $0.0 == .one}
    let data2 = labeledData.filter { $0.0 == .two}
    self.c1 = MultivariateGaussian(from: Tensor<Double>(data1.map { $0.1 }), given:p)
    self.c2 = MultivariateGaussian(from: Tensor<Double>(data2.map { $0.1 }), given:p)
  }
}

final class MonteCarloEMTests: XCTestCase {
  let data = [[1.0, 2.0], [1.0, 2.1], [6.0, 8.0], [6.2, 7.9]].map { Tensor<Double>($0)}
  
  /// Test low-variance resampling
  func testLowVarianceResampling() {
    enum Thingy {case a,b,c}
    var generator = ARC4RandomNumberGenerator(seed: 42)
    // make sure we have a bunch of small weight samples that will not be resampled
    let samples : [(Double,Thingy)] = [(0.1, .b), (60.0, .a),
                                       (0.1, .b), (0.1, .b),
                                       (90.0, .c), (0.1, .b)]
    // resample and check counts
    let resampled = resample(count:10, from:samples, using: &generator)
    var counts = Dictionary<Thingy,Int>()
    resampled.forEach { counts[$0, default: 0] += 1 }
    XCTAssertEqual(counts[.a, default: 0], 4)
    XCTAssertEqual(counts[.b, default: 0], 0)
    XCTAssertEqual(counts[.c, default: 0], 6)
    let _ = resample(count:10, from:samples) // make sure we can call default
  }
  
  /// Test fitting a simple 2-component mixture
  func testTwoComponents() {
    let generator = ARC4RandomNumberGenerator(seed: 11)
    var em = MonteCarloEM<TwoComponents>(sourceOfEntropy: generator)
    let model : TwoComponents = em.run(with:data, iterationCount: 5)
    assertEqual(model.c1.mean, Tensor<Double>([1.0, 2.0]), accuracy: 0.2)
    assertEqual(model.c2.mean, Tensor<Double>([6.0, 8.0]), accuracy: 0.2)
  }
}
