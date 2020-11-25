
@testable import SwiftFusion
import TensorFlow

import XCTest

final class PinholeCameraTests: XCTestCase {
  /// Tests constructor for identity pose.
  func testConstructorIdentityPose() {
    let K = Cal3_S2(fx: 100.0, fy: 100.0, s: 0.0, u0: 320.0, v0: 240.0)
    let camera = PinholeCamera(K)

    XCTAssertEqual(camera.calibration, K)
    XCTAssertEqual(camera.wTc, Pose3())
  }

  /// Tests default constructor.
  func testConstructorDefault() {
    let camera = PinholeCamera<Cal3_S2>()

    XCTAssertEqual(camera.calibration, Cal3_S2())
    XCTAssertEqual(camera.wTc, Pose3())
  }

  /// Tests project.
  func testProject() {
    let K = Cal3_S2(fx: 100.0, fy: 100.0, s: 0.0, u0: 320.0, v0: 240.0)

    // Camera 1 unit away looking into the the world origin, X axes are aligned
    let wTc = Pose3(
      Rot3(
        1, 0, 0,
        0, -1, 0,
        0, 0, -1), 
      Vector3(0, 0, 1))
    
    let camera = PinholeCamera(K, wTc)

    // A square of side 0.2 around the world origin
    let wp1 = Vector3(0.1, 0.1, 0.0)
    let wp2 = Vector3(0.1, -0.1, 0.0)
    let wp3 = Vector3(-0.1, -0.1, 0.0)
    let wp4 = Vector3(-0.1, 0.1, 0.0)

    // Manually compute the expected output. Example for wp1:
    // - Point in camera frame cp = (0.1, -0.1, 1.0)
    // - Project into normalized coordinate np = (0.1, -0.1)
    // - Uncalibrate to image coordinate ip = (330, 230)
    let ip1 = Vector2(330.0, 230.0)
    let ip2 = Vector2(330.0, 250.0)
    let ip3 = Vector2(310.0, 250.0)
    let ip4 = Vector2(310.0, 230.0)

    XCTAssertEqual(camera.project(wp1), ip1)
    XCTAssertEqual(camera.project(wp2), ip2)
    XCTAssertEqual(camera.project(wp3), ip3)
    XCTAssertEqual(camera.project(wp4), ip4)
  }

  /// Tests the custom derivative for project to normalized coordinate.
  func testPullbackProjectToNormalized() {
    let K = Cal3_S2(fx: 100.0, fy: 100.0, s: 0.0, u0: 320.0, v0: 240.0)
    let wTc = Pose3(
      Rot3(
        1, 0, 0,
        0, -1, 0,
        0, 0, -1), 
      Vector3(0, 0, 1))
    let camera = PinholeCamera(K, wTc)
    
    let wp = Vector3(1, 1, 0)

    let (p, pb) = valueWithPullback(at: camera, wp) { $0.projectToNormalized($1) }
    let dx = pb(Vector2(1, 0))
    let dy = pb(Vector2(0, 1))

    // Expected values computed by running through AD (disabling the custom derivative)
    let dxExpected = (
      PinholeCamera<Cal3_S2>.TangentVector(
        wTc: Vector6(-1.0, -2.0, -1.0, -1.0, 0.0, 1.0), 
        calibration: K.zeroTangentVector),
      Vector3(1.0, 0.0, 1.0))
    let dyExpected = (
      PinholeCamera<Cal3_S2>.TangentVector(
        wTc: Vector6(2.0, 1.0, -1.0, 0.0, -1.0, -1.0), 
        calibration: K.zeroTangentVector),
      Vector3(0.0, -1.0, -1.0))

    assertAllKeyPathEqual(p, Vector2(1.0, -1.0), accuracy: 1e-10)

    assertAllKeyPathEqual(dx.0.wTc, dxExpected.0.wTc, accuracy: 1e-10)
    XCTAssertEqual(dx.0.calibration, dxExpected.0.calibration)
    assertAllKeyPathEqual(dx.1, dxExpected.1, accuracy: 1e-10)
    
    assertAllKeyPathEqual(dy.0.wTc, dyExpected.0.wTc, accuracy: 1e-10)
    XCTAssertEqual(dy.0.calibration, dyExpected.0.calibration)
    assertAllKeyPathEqual(dy.1, dyExpected.1, accuracy: 1e-10)
  }

  /// Tests backproject
  func testBackproject() {
    let K = Cal3_S2(fx: 100.0, fy: 100.0, s: 0.0, u0: 320.0, v0: 240.0)

    // Camera 1 unit away looking into the the world origin, X axes are aligned
    let wTc = Pose3(
      Rot3(
        1, 0, 0,
        0, -1, 0,
        0, 0, -1), 
      Vector3(0, 0, 1))
    
    let camera = PinholeCamera(K, wTc)

    // A square of side 0.2 around the world origin
    let wp1 = Vector3(0.1, 0.1, 0.0)
    let wp2 = Vector3(0.1, -0.1, 0.0)
    let wp3 = Vector3(-0.1, -0.1, 0.0)
    let wp4 = Vector3(-0.1, 0.1, 0.0)

    // See test for project
    let ip1 = Vector2(330.0, 230.0)
    let ip2 = Vector2(330.0, 250.0)
    let ip3 = Vector2(310.0, 250.0)
    let ip4 = Vector2(310.0, 230.0)

    // Depth is 1 since the camera is 1 unit away
    XCTAssertEqual(camera.backproject(ip1, 1.0), wp1)
    XCTAssertEqual(camera.backproject(ip2, 1.0), wp2)
    XCTAssertEqual(camera.backproject(ip3, 1.0), wp3)
    XCTAssertEqual(camera.backproject(ip4, 1.0), wp4)
  }
}
