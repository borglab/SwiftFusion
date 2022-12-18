// Unit tests for Rot3 class, which models SO(3)
import _Differentiation
import SwiftFusion
// import TensorFlow

// Boilerplate code for running the test cases
import XCTest

final class Rot3Tests: XCTestCase {

  // test constructor from doubles
  func testConstructorEquality() {
    let R1 = Rot3()
    let R2 = Rot3(1, 0, 0,  0, 1, 0,  0, 0, 1)
    XCTAssertEqual(R1, R2)
  }

  // test constructor from doubles, non-identity
  func testConstructorInequal() {
    let R1 = Rot3()
    let R2 = Rot3(0, 1, 0, 1, 0, 0, 0, 0, -1)
    XCTAssertNotEqual(R1, R2)
  }

  /// Test if the conversion from quaternion to rotation matrix is correct
  func testQuaternionMatrix() {
    let R1 = Rot3.fromQuaternion(0.710997408193224, 0.360544029310185, 0.594459869568306, 0.105395217842782)
    let R2 = Rot3(
        0.271018623057411,   0.278786459830371,   0.921318086098018,
        0.578529366719085,   0.717799701969298,  -0.387385285854279,
       -0.769319620053772,   0.637998195662053,   0.033250932803219)

    let R3 = Rot3.fromQuaternion(0.263360579192421, 0.571813128030932, 0.494678363680335, 0.599136268678053);
    let R4 = Rot3(
        -0.207341903877828,   0.250149415542075,   0.945745528564780,
         0.881304914479026,  -0.371869043667957,   0.291573424846290,
         0.424630407073532,   0.893945571198514,  -0.143353873763946)
    
    assertAllKeyPathEqual(R1, R2, accuracy: 1e-5)
    assertAllKeyPathEqual(R3, R4, accuracy: 1e-5)
  }
  
  /// Tests that the manifold invariant holds for Rot3
  func testManifoldIdentity() {
    for _ in 0..<30 {
      let p = Rot3.fromTangent(Vector3(flatTensor: Tensor<Double>(randomNormal: [3])))
      let q = Rot3.fromTangent(Vector3(flatTensor: Tensor<Double>(randomNormal: [3])))
      let actual: Rot3 = Rot3(coordinate: p.coordinate.retract(p.coordinate.localCoordinate(q.coordinate)))
      assertAllKeyPathEqual(actual, q, accuracy: 1e-6)
    }
  }
  
  /// Tests that the manifold invariant holds for Rot3
  /// (-1+2n) * pi
  func testManifoldIdentitySpecial1() {
    for i in -5..<5 {
      let p = Rot3.fromTangent(Vector3(Double(2*i - 1) * .pi, 0, 0))
      let q = Rot3.fromTangent(Vector3(flatTensor: Tensor<Double>(randomNormal: [3])))
      let actual: Rot3 = Rot3(coordinate: p.coordinate.retract(p.coordinate.localCoordinate(q.coordinate)))
      assertAllKeyPathEqual(actual, q, accuracy: 1e-6)
    }
    
    for i in -5..<5 {
      let p = Rot3.fromTangent(Vector3(0, 0, Double(2*i - 1) * .pi))
      let q = Rot3.fromTangent(Vector3(flatTensor: Tensor<Double>(randomNormal: [3])))
      let actual: Rot3 = Rot3(coordinate: p.coordinate.retract(p.coordinate.localCoordinate(q.coordinate)))
      assertAllKeyPathEqual(actual, q, accuracy: 1e-6)
    }
  }
  
  /// Tests that the manifold invariant holds for Rot3
  func testManifoldIdentitySpecial2() {
    for _ in 0..<10 {
      let p = Rot3.fromTangent(Vector3(flatTensor: 1e-10 * Tensor<Double>(randomNormal: [3])))
      let q = Rot3.fromTangent(Vector3(flatTensor: Tensor<Double>(randomNormal: [3])))
      let actual: Rot3 = Rot3(coordinate: p.coordinate.retract(p.coordinate.localCoordinate(q.coordinate)))
      assertAllKeyPathEqual(actual, q, accuracy: 1e-6)
    }
  }
  
  /// Tests that the manifold invariant holds for Rot3
  /// (-1+2n) * pi
  func testManifoldIdentitySpecial3() {
    for i in -5..<5 {
      let p = Rot3.fromTangent(Vector3(Double(2*i - 1) * .pi, 0, 0))
      let q = Rot3.fromTangent(Vector3(0, 0, 0))
      let actual: Rot3 = Rot3(coordinate: p.coordinate.retract(p.coordinate.localCoordinate(q.coordinate)))
      assertAllKeyPathEqual(actual, q, accuracy: 1e-6)
    }
    
    for i in -5..<5 {
      let p = Rot3.fromTangent(Vector3(0, 0, Double(2*i - 1) * .pi))
      let q = Rot3.fromTangent(Vector3(0, 0, 0))
      let actual: Rot3 = Rot3(coordinate: p.coordinate.retract(p.coordinate.localCoordinate(q.coordinate)))
      assertAllKeyPathEqual(actual, q, accuracy: 1e-6)
    }
  }
  
  // test that move really works
  func testMove() {
    let xi = Vector3(-.pi/2,0,0)
    var actual = Rot3()
    actual.move(along: xi)
    let expected = Rot3(coordinate:
      Matrix3Coordinate(
        1, 0, 0,
        0, 0, 1,
        0, -1, 0)
    )
    assertAllKeyPathEqual(actual, expected, accuracy: 1e-6)
  }
  
  func testExpmap() {
    let axis = Vector3(0, 1, 0)  // rotation around Y
    let angle: Double = 3.14 / 4.0
    let v = angle * axis
    let expected = Matrix3(0.707388, 0, 0.706825, 0, 1, 0, -0.706825, 0, 0.707388)
    
    var actual = Rot3()
    actual.move(along: v)
    
    assertAllKeyPathEqual(actual.coordinate.R, expected, accuracy: 1e-5)
  }
  
  func testExpmapNearZero() {
    let axis = Vector3(0, 1, 0)  // rotation around Y
    let angle: Double = 0.0
    let v = angle * axis
    let expected = Rot3()
    
    var actual = Rot3()
    actual.move(along: v)
    
    assertAllKeyPathEqual(actual, expected, accuracy: 1e-5)
  }

  /// Tests that the custom implementations of `Adjoint` and `AdjointTranspose` are correct.
  func testAdjoint() {
    for _ in 0..<10 {
      let rot = Rot3.fromTangent(Vector3(flatTensor: Tensor<Double>(randomNormal: [3])))
      for v in Pose2.TangentVector.standardBasis {
        assertEqual(
          rot.Adjoint(v).flatTensor,
          rot.coordinate.defaultAdjoint(v).flatTensor,
          accuracy: 1e-10
        )
        assertEqual(
          rot.AdjointTranspose(v).flatTensor,
          rot.coordinate.defaultAdjointTranspose(v).flatTensor,
          accuracy: 1e-10
        )
      }
    }
  }
  
  /// Tests the ClosestTo function.
  func testClosestTo() {
    let M = Matrix3(
          0.79067393, 0.6051136, -0.0930814,
          0.4155925, -0.64214347, -0.64324489,
          -0.44948549, 0.47046326, -0.75917576
    )

    let expected = Matrix3(
          0.790687, 0.605096, -0.0931312,
          0.415746, -0.642355, -0.643844,
          -0.449411, 0.47036, -0.759468
    )

    let actual = Rot3.ClosestTo(mat: 3 * M).coordinate.R
    assertAllKeyPathEqual(expected, actual, accuracy: 1e-6)
  }
  
  /// Tests that our derivatives will not fail when the rotation has slightly drifted away from the SO(3) manifold
  func testExtreme() {
    let R1 = Rot3()
    
    let R2 = Rot3.fromTangent(Vector3(0,0, .pi-0.01))
    
    let R2_drifted = Rot3(
      -0.9999500004166653, -0.009999833334166574, 0.0,
      0.009999833334166574, -0.9999500004166653, 0.0,
      0.0, 0.0, 0.9999
    )
    
    let diff = R1.localCoordinate(R2_drifted)
    // First ensure we don't get NaNs
    XCTAssert(!diff.x.isNaN)
    
    let diff_normal = R1.localCoordinate(R2)
    assertAllKeyPathEqual(diff, diff_normal, accuracy: 1e-2)
  }

  // Check group action: basic rotations
  func testRotateBasic() {
    let p = Vector3(2.0, -4.0, 6.0)
    let Rx = Rot3.fromTangent(Vector3(60.0 * .pi / 180.0, 0.0, 0.0))
    let Ry = Rot3.fromTangent(Vector3(0.0, -30.0 * .pi / 180.0, 0.0))
    let Rz = Rot3.fromTangent(Vector3(0.0, 0.0, 150.0 * .pi / 180.0))

    // Basic rotations along coordinate axes
    let expectedRx = Vector3(2.0, -2.0 - 3.0 * .sqrt(3.0), -2.0 * .sqrt(3.0) + 3.0)
    let expectedRy = Vector3(.sqrt(3.0) - 3.0, -4.0, 1.0 + 3.0 * .sqrt(3.0))
    let expectedRz = Vector3(-.sqrt(3.0) + 2.0, 1.0 + 2.0 * .sqrt(3.0), 6.0)

    let actualRx1 = Rx.rotate(p)
    let actualRx2 = Rx * p
    let actualRy1 = Ry.rotate(p)
    let actualRy2 = Ry * p
    let actualRz1 = Rz.rotate(p)
    let actualRz2 = Rz * p

    assertAllKeyPathEqual(actualRx1, expectedRx, accuracy: 1e-9)
    assertAllKeyPathEqual(actualRx2, expectedRx, accuracy: 1e-9)
    assertAllKeyPathEqual(actualRy1, expectedRy, accuracy: 1e-9)
    assertAllKeyPathEqual(actualRy2, expectedRy, accuracy: 1e-9)
    assertAllKeyPathEqual(actualRz1, expectedRz, accuracy: 1e-9)
    assertAllKeyPathEqual(actualRz2, expectedRz, accuracy: 1e-9)
  }

  // Check group action: rotation along arbitrary axis
  func testRotateArbitrary() {
    let p = Vector3(1.0, -2.0, 3.0)
    let R = Rot3.fromTangent(Vector3(0.1, 0.2, -0.3))

    // Expected value generated using the following Python script:
    // import numpy as np
    // import cv2
    // np.set_printoptions(precision=16)
    // R = cv2.Rodrigues(np.array([0.1, 0.2, -0.3]))[0]
    // R.dot(np.array([1.0, -2.0, 3.0]))
    let expected = Vector3(0.871509606555838, -2.5663299211301474, 2.5796165880985145)
    let actual1 = R.rotate(p)
    let actual2 = R * p

    assertAllKeyPathEqual(actual1, expected, accuracy: 1e-9)
    assertAllKeyPathEqual(actual2, expected, accuracy: 1e-9)
  }

  // Check group action: unrotate
  func testUnrotate() {
    let p = Vector3(1.0, -2.0, 3.0)
    let R = Rot3.fromTangent(Vector3(0.1, 0.2, -0.3))

    // Should be identity
    let actual = R.unrotate(R.rotate(p))

    assertAllKeyPathEqual(actual, p, accuracy: 1e-9)
  }
}
