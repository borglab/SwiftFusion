import XCTest
import PenguinStructures
import SwiftFusion
import TensorFlow

class RandomProjectionTests: XCTestCase {
  func testEncode() {
    let image = Tensor<Double>(randomNormal: [20, 30, 1])
    let d = 5
    let projector = RandomProjection(fromShape: image.shape, toFeatureSize: d)
    let features = projector.encode(image)

    XCTAssertEqual(features.rank, 1)
    XCTAssertEqual(features.shape[0], 5)
  }

  func testEncodeBatch() {
    let image = Tensor<Double>(randomNormal: [3, 20, 30, 1])
    let d = 5
    let projector = RandomProjection(fromShape: [20, 30, 1], toFeatureSize: d)


    XCTAssertEqual(projector.B.shape, TensorShape([5, 20*30*1]))

    let features = projector.encode(image)

    XCTAssertEqual(features.rank, 2)
    XCTAssertEqual(features.shape[0], 3)
    XCTAssertEqual(features.shape[1], 5)
  }
}