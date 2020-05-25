// Unit tests for Rot3 class, which models SO(3)
import SwiftFusion
import TensorFlow

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
      let p = Rot3.fromTangent(Vector3(Tensor<Double>(randomNormal: [3])))
      let q = Rot3.fromTangent(Vector3(Tensor<Double>(randomNormal: [3])))
      let actual: Rot3 = Rot3(coordinate: p.coordinate.retract(p.coordinate.localCoordinate(q.coordinate)))
      assertAllKeyPathEqual(actual, q, accuracy: 1e-10)
    }
  }
  
  /// Tests that the manifold invariant holds for Rot3
  /// (-1+2n) * pi
  func testManifoldIdentitySpecial1() {
    for i in -5..<5 {
      let p = Rot3.fromTangent(Vector3(Double(2*i - 1) * .pi, 0, 0))
      let q = Rot3.fromTangent(Vector3(Tensor<Double>(randomNormal: [3])))
      let actual: Rot3 = Rot3(coordinate: p.coordinate.retract(p.coordinate.localCoordinate(q.coordinate)))
      assertAllKeyPathEqual(actual, q, accuracy: 1e-10)
    }
    
    for i in -5..<5 {
      let p = Rot3.fromTangent(Vector3(0, 0, Double(2*i - 1) * .pi))
      let q = Rot3.fromTangent(Vector3(Tensor<Double>(randomNormal: [3])))
      let actual: Rot3 = Rot3(coordinate: p.coordinate.retract(p.coordinate.localCoordinate(q.coordinate)))
      assertAllKeyPathEqual(actual, q, accuracy: 1e-10)
    }
  }
  
  /// Tests that the manifold invariant holds for Rot3
  func testManifoldIdentitySpecial2() {
    for _ in 0..<10 {
      let p = Rot3.fromTangent(Vector3(1e-10 * Tensor<Double>(randomNormal: [3])))
      let q = Rot3.fromTangent(Vector3(Tensor<Double>(randomNormal: [3])))
      let actual: Rot3 = Rot3(coordinate: p.coordinate.retract(p.coordinate.localCoordinate(q.coordinate)))
      assertAllKeyPathEqual(actual, q, accuracy: 1e-10)
    }
  }
  
  /// Tests that the manifold invariant holds for Rot3
  /// (-1+2n) * pi
  func testManifoldIdentitySpecial3() {
    for i in -5..<5 {
      let p = Rot3.fromTangent(Vector3(Double(2*i - 1) * .pi, 0, 0))
      let q = Rot3.fromTangent(Vector3(0, 0, 0))
      let actual: Rot3 = Rot3(coordinate: p.coordinate.retract(p.coordinate.localCoordinate(q.coordinate)))
      assertAllKeyPathEqual(actual, q, accuracy: 1e-10)
    }
    
    for i in -5..<5 {
      let p = Rot3.fromTangent(Vector3(0, 0, Double(2*i - 1) * .pi))
      let q = Rot3.fromTangent(Vector3(0, 0, 0))
      let actual: Rot3 = Rot3(coordinate: p.coordinate.retract(p.coordinate.localCoordinate(q.coordinate)))
      assertAllKeyPathEqual(actual, q, accuracy: 1e-10)
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
    assertAllKeyPathEqual(actual, expected, accuracy: 1e-8)
  }
  
  func testExpmap() {
    let axis = Vector3(0, 1, 0)  // rotation around Y
    let angle = 3.14 / 4.0
    let v = angle * axis
    let expected = Matrix3(0.707388, 0, 0.706825, 0, 1, 0, -0.706825, 0, 0.707388)
    
    var actual = Rot3()
    actual.move(along: v)
    
    assertAllKeyPathEqual(actual.coordinate.R, expected, accuracy: 1e-5)
  }
  
  func testExpmapNearZero() {
    let axis = Vector3(0, 1, 0)  // rotation around Y
    let angle = 0.0
    let v = angle * axis
    let expected = Rot3()
    
    var actual = Rot3()
    actual.move(along: v)
    
    assertAllKeyPathEqual(actual, expected, accuracy: 1e-5)
  }

  /// Tests that the custom implementations of `Adjoint` and `AdjointTranspose` are correct.
  func testAdjoint() {
    for _ in 0..<10 {
      let rot = Rot3.fromTangent(Vector3(Tensor<Double>(randomNormal: [3])))
      for v in Pose2.TangentVector.standardBasis {
        assertEqual(
          rot.Adjoint(v).tensor,
          rot.coordinate.defaultAdjoint(v).tensor,
          accuracy: 1e-10
        )
        assertEqual(
          rot.AdjointTranspose(v).tensor,
          rot.coordinate.defaultAdjointTranspose(v).tensor,
          accuracy: 1e-10
        )
      }
    }
  }
}
