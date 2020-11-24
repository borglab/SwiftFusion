
import SwiftFusion
import TensorFlow

import XCTest

final class Cal3_S2Tests: XCTestCase {
  /// Tests default constructor.
  func testConstructorDefault() {
    let K1 = Cal3_S2()
    let K2 = Cal3_S2(fx: 1.0, fy: 1.0, s: 0.0, u0: 0.0, v0: 0.0)
    XCTAssertEqual(K1, K2)
  }

  /// Tests uncalibrate.
  func testCalibrateUncalibrate() {
    let K = Cal3_S2(fx: 200.0, fy: 200.0, s: 1.0, u0: 320.0, v0: 240.0)
    let np = Vector2(1.0, 2.0)
    
    let expected = Vector2(522.0, 640.0)  // Manually calculated

    XCTAssertEqual(K.uncalibrate(np), expected)
  }

  /// Tests calibrate identity.
  func testCalibrateIdentity() {
    let K = Cal3_S2(fx: 200.0, fy: 200.0, s: 1.0, u0: 320.0, v0: 240.0)
    let np = Vector2(1.0, 2.0)

    XCTAssertEqual(K.calibrate(K.uncalibrate(np)), np)
  }

  /// Tests manifold.
  func testManifold() {
    var K1 = Cal3_S2(fx: 200.0, fy: 200.0, s: 1.0, u0: 320.0, v0: 240.0)
    let K2 = Cal3_S2(fx: 201.0, fy: 202.0, s: 4.0, u0: 324.0, v0: 245.0)
    let dK = Vector5(1.0, 2.0, 3.0, 4.0, 5.0)

    XCTAssertEqual(K1.retract(dK), K2)
    XCTAssertEqual(K1.localCoordinate(K2), dK)

    K1.move(along: dK)
    
    XCTAssertEqual(K1, K2)
  }
}
