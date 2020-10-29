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

import TensorFlow
import XCTest

import BeeTracking

class BeeTrackingTests: XCTestCase {
  /// Test that the hand-coded Jacobian for the decode method gives the same results as the
  /// AD-generated Jacobian.
  func testDecodeJacobian() {
    // Size of the images.
    let h = 10
    let w = 10
    let c = 10

    // Number of batches to test. (`decodeJacobian` currently only supports one batch at a time).
    let batchCount = 1

    // Model size parameters.
    let latentDimension = 5
    let hiddenDimension = 100

    // A random model and a random latent code to decode.
    let model = DenseRAE(
      imageHeight: h, imageWidth: w, imageChannels: c,
      hiddenDimension: hiddenDimension, latentDimension: latentDimension)
    let latent = Tensor<Double>(randomNormal: [batchCount, latentDimension])

    // The hand-coded Jacobian.
    let actualJacobian = model.decodeJacobian(latent)

    // The AD-generated pullback function.
    let pb = pullback(at: latent) { model.decode($0) }

    // Pass all the unit vectors throught the AD-generated pullback function and check that the
    // results match the hand-coded Jacobian.
    for batch in 0..<batchCount {
      for i in 0..<h {
        for j in 0..<w {
          for k in 0..<c {
            var unit = Tensor<Double>(zeros: [batchCount, h, w, c])
            unit[batch, i, j, k] = Tensor(1)
            XCTAssertLessThan(
              (actualJacobian[batch, i, j, k] - pb(unit)).squared().sum().scalar!, 1e-6)
          }
        }
      }
    }
  }
}
