import XCTest
import SwiftFusion
import TensorFlow

final class Pose3Tests: XCTestCase {
  /// Tests that the manifold invariant holds for Pose2
  func testManifoldIdentity() {
    for _ in 0..<10 {
      let p = Pose3.fromTangent(Vector6(Tensor<Double>(randomNormal: [6])))
      let q = Pose3.fromTangent(Vector6(Tensor<Double>(randomNormal: [6])))
      let actual: Pose3 = Pose3(coordinate: p.coordinate.retract(p.coordinate.localCoordinate(q.coordinate)))
      assertAllKeyPathEqual(actual, q, accuracy: 1e-10)
    }
  }
  
  func testTrivialRotation() {
    let p = Pose3.fromTangent(Vector6(Tensor<Double>([0.3, 0, 0, 0, 0, 0])))
    let R = Rot3.fromTangent(Vector3(0.3, 0, 0))
    
    assertAllKeyPathEqual(p.rot, R, accuracy: 1e-10)
  }
  
  func testTrivialFull1() {
    let P = Vector3(0.2,0.7,-2)
    let R = Rot3.fromTangent(Vector3(0.3, 0, 0))
    let p = Pose3.fromTangent(Vector6(Tensor<Double>([0.3, 0, 0, 0.2, 0.394742, -2.08998])))
    assertAllKeyPathEqual(p.rot, R, accuracy: 1e-10)
    assertAllKeyPathEqual(p.t, P, accuracy: 1e-5)
  }
  
  func testTrivialFull2() {
    let P = Vector3(3.5,-8.2,4.2)
    let R = Rot3.fromTangent(Vector3(0.3, 0, 0))
    let t12 = Vector6(Tensor<Double>(repeating: 0.1, shape: [6]))
    let t1 = Pose3(R, P)
    let t2 = Pose3(coordinate: t1.coordinate.retract(t12))
    assertAllKeyPathEqual(t1.coordinate.localCoordinate(t2.coordinate), t12, accuracy: 1e-5)
  }
}
