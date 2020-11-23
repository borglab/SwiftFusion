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

import ModelSupport
import SwiftFusion
import TensorFlow
import XCTest

final class PatchTests: XCTestCase {
  /// A patch at a non-rotated bounding box is the same as a slice of the image.
  func testSlice() {
    let image = Tensor<Double>(randomUniform: [100, 100, 3])
    let patch = image.patch(
      at: OrientedBoundingBox(center: Pose2(Rot2(0), Vector2(60, 30)), rows: 10, cols: 20))
    let expectedPatch = image.slice(lowerBounds: [25, 50, 0], sizes: [10, 20, 3])
    XCTAssertEqual(patch, expectedPatch)
  }

  // /// The derivative of a patch with respect to the input image is the identity restricted to the
  // /// patch region.
  // func testSliceDerivativeWithRespectToImage() {
  //   let image = Tensor<Double>(randomUniform: [100, 100, 3])
  //   let grad = gradient(at: image) { image in
  //     image.patch(
  //       at: OrientedBoundingBox(center: Pose2(Rot2(0), Vector2(60, 30)), rows: 10, cols: 20)
  //     ).sum()
  //   }
  //   var expectedGrad = Tensor<Double>(zeros: [100, 100, 3])
  //   expectedGrad[25..<35, 50..<70] = Tensor(ones: [10, 20, 3])
  //   XCTAssertEqual(grad, expectedGrad)
  // }

  /// Test cropping an example image.
  func testExampleImage() {
    let dataDir = URL.sourceFileDirectory().appendingPathComponent("data")
    let image = Tensor<Double>(
      Image(contentsOf: dataDir.appendingPathComponent("test.png")).tensor)
    let obb = OrientedBoundingBox(
      center: Pose2(Rot2(-20 * .pi / 180), Vector2(35, 65)), rows: 20, cols: 40)
    let patch = image.patch(at: obb)

    // Created using ImageMagick:
    //   convert -distort SRT 35,65,20 -crop 40x20+15+55 test.png cropped.png
    let expectedPatch =
      Tensor<Double>(Image(contentsOf: dataDir.appendingPathComponent("cropped.png")).tensor)

    // The actual and expected are pretty close, but not precisely the same.
    // TODO: Investigate the difference.
    XCTAssertLessThan((patch - expectedPatch).max().scalarized(), 40)
    XCTAssertLessThan(sqrt((patch - expectedPatch).squared().mean().scalarized()), 10)
  }

  /// The derivative of a patch with respect to the region's position is 0 when the image is
  /// constant.
  func testDerivativeWithRespectToPosition_constantImage() {
    let image = Tensor<Double>(ones: [100, 100, 3])
    let obb = OrientedBoundingBox(center: Pose2(Rot2(1), Vector2(60, 50)), rows: 10, cols: 20)
    let grad = gradient(at: obb) { obb in
      image.patch(at: obb).mean()
    }
    XCTAssertEqual(grad.center.norm, 0, accuracy: 1e-6)
  }

  /// The derivative of a patch with respect to the region's position is equal to the slope of
  /// an image that increases linearly from one side to the other.
  func testDerivativeWithRespectToPosition_slopeImage() {
    let image = Tensor(linearSpaceFrom: 0.01, to: 1, count: 100)
      .reshaped(to: [1, 100, 1])
      .tiled(multiples: [100, 1, 1])
    let obb = OrientedBoundingBox(center: Pose2(Rot2(0), Vector2(60, 50)), rows: 10, cols: 20)
    let grad = gradient(at: obb) { obb in
      image.patch(at: obb).mean()
    }
    let expectedGrad = Vector3(0, 0.01, 0.00)
    XCTAssertEqual((grad.center - expectedGrad).norm, 0, accuracy: 1e-6)
  }

  /// Test that we can gradient descend on the position to reach a target region in the image.
  func testGradientDescentPosition() {
    // A target patch whose position we want to find.
    let target = Tensor(linearSpaceFrom: 1, to: 2, count: 10)
      .reshaped(to: [1, 10, 1])
      .tiled(multiples: [5, 1, 1])

    // An image with zeros everywhere except the target.
    var image = Tensor<Double>(zeros: [100, 100, 1])
    image[20..<25, 60..<70] = target

    // An initial guess that has some overlap with the target.
    var x = OrientedBoundingBox(center: Pose2(Rot2(1.0), Vector2(63, 22)), rows: 5, cols: 10)

    // Use gradient descent to find an oriented bounding box containing all ones.
    let stepCount = 100
    for _ in 0..<stepCount {
      let g = gradient(at: x) { x in
        (image.patch(at: x) - target).squared().mean()
      }
      x.center.move(along: -0.2 * g.center)
    }

    let expectedCenter = Pose2(Rot2(0), Vector2(65, 22.5))
    XCTAssertLessThan(expectedCenter.localCoordinate(x.center).norm, 0.1)
  }

  /// Test that the result and jacobian of `patchWithJacobian` are the same as the result and
  /// jacobian that we get from using autodiff on `patch`.
  func testPatchWithJacobian() {
    for _ in 0..<10 {
      let image = Tensor<Double>(randomNormal: [100, 100, 1])
      let obb = OrientedBoundingBox(
        center: Pose2(randomWithCovariance: eye(rowCount: 3)), rows: 10, cols: 20)

      // Calculate the expected value and jacobian using autodiff. We calculate the jacobian by
      // pulling back all the basis vectors.
      let (expectedValue, pb) = valueWithPullback(at: obb) { image.patch(at: $0) }
      let expectedJacobian = Tensor<Double>(
        stacking: (0..<(obb.rows * obb.cols)).map { i in
          var basis = Tensor<Double>(zerosLike: expectedValue)
          basis[i / obb.cols, i % obb.cols, 0] = Tensor(1)
          return pb(basis).center.flatTensor
        }
      ).reshaped(to: [obb.rows, obb.cols, 1, Pose2.TangentVector.dimension])

      // Compare to the result from `patchWithJacobian`.
      let (value, jacobian) = image.patchWithJacobian(at: obb)
      assertEqual(value, expectedValue, accuracy: 1e-6)
      assertEqual(jacobian, expectedJacobian, accuracy: 1e-6)
    }
  }
}
