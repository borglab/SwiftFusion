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

/// This file tests Monte Carlo EM training of appearance models

import SwiftFusion
import TensorFlow
import XCTest

// A likelihood model has a foreground and background model, as well as a feature encoder
struct LikelihoodModel<Encoder: AppearanceModelEncoder, FG:GenerativeDensity, BG:GenerativeDensity> {
  /// A likelihood model is trained on image patches
  let encoder: Encoder
  let fg: FG
  let bg: BG
  
  public init(encoder: Encoder, fg: FG,  bg: BG) {
    self.encoder = encoder
    self.fg = fg
    self.bg = bg
  }
  
  /// Train from a collection of image patches
  public init(from imageBatch: Tensor<Double>) {
    self.encoder = Encoder(from: imageBatch)
    self.fg = FG(from: encoder.encode(imageBatch))
    self.bg = BG(from: encoder.encode(imageBatch))
  }
}

// Make it trainable with EM
extension LikelihoodModel : McEmModel {
  /// As datum we have a (giant) image and a noisy manual label for an image patch
  typealias Datum = (frame: Tensor<Double>, manualLabel:OrientedBoundingBox)
  
  /// As hidden variable we use the "true" pose of the patch
  typealias Hidden = Pose2
  
  /// Initialize to uninitialized components
  public init(_ data:[Datum], using sourceOfEntropy: inout AnyRandomNumberGenerator) {
    // We do an initial training of the encode with the manually labeled images
    let paches = data.map { $0.frame.patch(at: $0.manualLabel) }
    let imageBatch = Tensor<Double>(paches)
    self.init(from: imageBatch)
  }
  
  /// Given a datum and a model, sample from the hidden variables
  func sample(count:Int, for datum: Datum,
              using sourceOfEntropy: inout AnyRandomNumberGenerator) -> [Hidden] {
    let labels = [Hidden]()
    //    let labels : [Hidden] = (0..<count).map { _ in
    //      let u = Double.random(in: 0..<p1+p2, using: &sourceOfEntropy)
    //      return u<=p1 ? .one : .two
    //    }
    return labels
  }
  
  /// Given an array of labeled datums, fit the two Gaussian mixture components
  mutating func fit(_ labeledData: [LabeledDatum]) {
  }
}

// Test with a random projection feature space, and Gaussian/NB for FG/BG
typealias RPGaussianNB = LikelihoodModel<RandomProjection,MultivariateGaussian, GaussianNB>

final class LikelihoodModelEMTests: XCTestCase {
    let deterministicEntropy = ARC4RandomNumberGenerator(seed: 42)
  
  /// Test fitting a simple 2-component mixture
  func testLikelihoodModel() {
    let frame = Tensor<Double>(zeros:[1000,1000,1])
    let data = [Vector2(100, 200), Vector2(150, 201), Vector2(600, 800)].map {
      (frame, OrientedBoundingBox(center: Pose2(Rot2(0), $0), rows: 70, cols: 40))
    }
    var em = MonteCarloEM<RPGaussianNB>(sourceOfEntropy: deterministicEntropy)
    let model : LikelihoodModel = em.run(with:data, iterationCount: 5)
  }
}
