import XCTest

import TensorFlow

import SwiftFusion

class AnyDifferentiableTests: XCTestCase {
  /// Tests that we can take derivatives with respect to an array of type-erased elements with
  /// different runtime types.
  func testAnyDifferentiableArray() {
    func f(_ poses: [Pose2], _ vectors: [Vector2]) -> Double {
      var loss: Double = 0
      for i in withoutDerivative(at: 0..<poses.count-1) {
        let b = between(poses[i], poses[i + 1])
        loss += b.t.norm
        loss += b.rot.theta * b.rot.theta
      }
      for i in withoutDerivative(at: 0..<vectors.count-1) {
        let b = vectors[i + 1] - vectors[i]
        loss += b.norm
      }
      return loss
    }

    func fErased(_ erased: [AnyDifferentiable]) -> Double {
      let poses = [erased[0], erased[1], erased[2]].differentiableMap { $0.baseAs(Pose2.self) }
      let vectors = [erased[3], erased[4], erased[5]].differentiableMap { $0.baseAs(Vector2.self) }
      return f(poses, vectors)
    }

    let initialPoses = (0..<3).map { _ in Pose2(randomWithCovariance: eye(rowCount: 3)) }
    let initialVectors = (0..<3).map { _ in Vector2(Tensor(randomNormal: [2])) }
    let elementsErased =
      initialPoses.map { AnyDifferentiable($0) } + initialVectors.map { AnyDifferentiable($0) }

    let (expectedPosesGrad, expectedVectorsGrad) = gradient(at: initialPoses, initialVectors, in: f)
    let erasedActualGrad = gradient(at: elementsErased, in: fErased)
    let actualPosesGrad = erasedActualGrad[0..<3].map { $0.base as! Pose2.TangentVector }
    let actualVectorsGrad = erasedActualGrad[3..<6].map { $0.base as! Vector2.TangentVector }
    XCTAssertEqual(actualPosesGrad, expectedPosesGrad.base)
    XCTAssertEqual(actualVectorsGrad, expectedVectorsGrad.base)
  }

  static var allTests = [
    ("testAnyDifferentiableArray", testAnyDifferentiableArray)
  ]
}
