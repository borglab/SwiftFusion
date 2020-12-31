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

import BeeTracking
import SwiftFusion
import TensorFlow
import XCTest

// Test with a random projection feature space, and Gaussian/NB for FG/BG
typealias RPGaussianNB = TrackingLikelihoodModel<RandomProjection,MultivariateGaussian, GaussianNB>

final class TrackingLikelihoodModelTests: XCTestCase {
  /// Test fitting a simple 2-component mixture
  func testTrackingLikelihoodModel() {
    let frame = Tensor<Double>(zeros:[1000,1000,1])
    let boundingBoxes = [Vector2(100, 200), Vector2(150, 201), Vector2(600, 800)].map {
      OrientedBoundingBox(center: Pose2(Rot2(0), $0), rows: 70, cols: 40)
    }
    let patches = Tensor<Double>(boundingBoxes.map {obb in frame.patch(at: obb)})
    let model = RPGaussianNB(from:patches, and:patches)
    XCTAssertEqual(model.encoder.B.shape, [5,70*40])
    XCTAssertEqual(model.foregroundModel.mean.shape, [5])
    XCTAssertEqual(model.backgroundModel.mu.shape, [5])
  }
}

final class TrackingLikelihoodModelEMTests: XCTestCase {
  /// Test fitting a simple 2-component mixture
  func testTrackingLikelihoodModel() {
    let generator = ARC4RandomNumberGenerator(seed: 42)
    let frame = Tensor<Double>(zeros:[1000,1000,1])
    let data = [(Vector2(100, 200), RPGaussianNB.PatchType.fg),
                (Vector2(150, 201), RPGaussianNB.PatchType.fg),
                (Vector2(600, 800), RPGaussianNB.PatchType.bg)].map {
      (frame, $1, OrientedBoundingBox(center: Pose2(Rot2(0), $0), rows: 70, cols: 40))
    }
    var em = MonteCarloEM<RPGaussianNB>(sourceOfEntropy: generator)
    let model = em.run(with:data, iterationCount: 3)
    XCTAssertEqual(model.encoder.B.shape, [5,70*40])
    XCTAssertEqual(model.foregroundModel.mean.shape, [5])
    XCTAssertEqual(model.backgroundModel.mu.shape, [5])
  }
}
