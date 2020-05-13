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
    let v = axis.scaled(by: angle)
    let expected = Tensor<Double>([[ 0.707388, 0, 0.706825 ], [0, 1, 0], [-0.706825, 0, 0.707388]])
    
    var actual = Rot3()
    actual.move(along: v)
    
    assertEqual(actual.coordinate.R, expected, accuracy: 1e-5)
  }
  
  func testExpmapNearZero() {
    let axis = Vector3(0, 1, 0)  // rotation around Y
    let angle = 0.0
    let v = axis.scaled(by: angle)
    let expected = Rot3()
    
    var actual = Rot3()
    actual.move(along: v)
    
    assertAllKeyPathEqual(actual, expected, accuracy: 1e-5)
  }
}
