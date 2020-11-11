// Copyright 2020 The SwiftFusion Authors. All Rights Reserved.
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

import Foundation
import TensorFlow
import SwiftFusion
import XCTest

final class ProbablisticTrackingFactorTests: XCTestCase {
  func testCreation() {
    var v = VariableAssignments()
    let poseId = v.store(Pose2())

    let _ = ProbablisticTrackingFactor(poseId,
      measurement: Tensor<Double>([[0]]),
      encoder: PPCA(latentSize: 10),
      patchSize: (10, 10),
      appearanceModelSize: (10, 1),
      foregroundModel: GaussianNB(dims: [10]),
      backgroundModel: GaussianNB(dims: [10])
    )
  }

  /// Test sanity of the error by a simple case
  func testErrorSanity() {
    var v = VariableAssignments()
    let pose = Pose2(4.5, 4.5, 0)
    let poseId = v.store(pose)

    /// Make a center one image
    var image: Tensor<Double> = .init(zeros: [8, 8, 3])

    image[4, 4, 0] = Tensor(1.0)

    let featureDim = 10
    var encoder = PPCA(latentSize: featureDim)

    /// Encoder with only center activation of 1
    encoder.W = Tensor<Double>(zeros: [3, 3, 3, 10])
    var W_inv = Tensor<Double>(zeros: [10, 3, 3, 3])
    encoder.mu = Tensor<Double>(zeros: [3, 3, 3])

    encoder.W[1, 1, 0, 0] = Tensor(1.0)
    W_inv[0, 1, 1, 0] = Tensor(1.0)
    encoder.W_inv = W_inv.reshaped(to: [10, 3 * 3 * 3])

    /// Training foreground and background models
    var fg_model = GaussianNB(dims: [featureDim], regularizer: 1e-8)
    var bg_model = GaussianNB(dims: [featureDim], regularizer: 1e-8)
    
    var sample_fg = Tensor<Double>(zeros: [2, 10])
    var sample_bg = Tensor<Double>(zeros: [2, 10])

    sample_fg[0, 0] = Tensor(1.1)
    sample_fg[1, 0] = Tensor(0.9)

    sample_bg[0, 2] = Tensor(1.1)
    sample_bg[1, 2] = Tensor(0.9)

    fg_model.fit(sample_fg)
    bg_model.fit(sample_bg)

    let factor = ProbablisticTrackingFactor(poseId,
      measurement: image,
      encoder: encoder,
      patchSize: (3, 3),
      appearanceModelSize: (3, 3),
      foregroundModel: fg_model,
      backgroundModel: bg_model
    )
    
    /// Check if we have the desired error at minima
    assertAllKeyPathEqual(factor.errorVector(pose), Vector1(-5000000000000047.0), accuracy: 1e-1)
  }
}
