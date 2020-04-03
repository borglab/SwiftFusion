import TensorFlow
import XCTest

import SwiftFusion

final class Pose2Tests: XCTestCase {
  /// test between for trivial values
  func testBetweenIdentitiesTrivial() {
    let wT1 = Pose2(0, 0, 0), wT2 = Pose2(0, 0, 0)
    let expected = Pose2(0, 0, 0)
    let actual = between(wT1, wT2)
    XCTAssertEqual(actual, expected)
  }

  /// test between function for non-rotated poses
  func testBetweenIdentities() {
    let wT1 = Pose2(2, 1, 0), wT2 = Pose2(5, 2, 0)
    let expected = Pose2(3, 1, 0)
    let actual = between(wT1, wT2)
    XCTAssertEqual(actual, expected)
  }

  /// test between function for rotated poses
  func testBetweenIdentitiesRotated() {
    let wT1 = Pose2(1, 0, 3.1415926 / 2.0), wT2 = Pose2(1, 0, -3.1415926 / 2.0)
    let expected = Pose2(0, 0, -3.1415926)
    let actual = between(wT1, wT2)
    // dump(expected, name: "expected");
    // dump(actual, name: "actual");
    XCTAssertEqual(actual, expected)
  }

  /// test the simplest gradient descent on Pose2
  func testBetweenDerivatives() {
    var pT1 = Pose2(Rot2(0), Vector2(1, 0)), pT2 = Pose2(Rot2(1), Vector2(1, 1))

    for _ in 0..<100 {
      let (_, 𝛁loss) = valueWithGradient(at: pT1) { pT1 -> Double in
        var loss: Double = 0
        let ŷ = between(pT1, pT2)
        let error = ŷ.rot.theta * ŷ.rot.theta + ŷ.t.x * ŷ.t.x + ŷ.t.y * ŷ.t.y
        loss = loss + (error / 10)

        return loss
      }

      // print("𝛁loss", 𝛁loss)
      pT1.move(along: 𝛁loss.scaled(by: -1))
    }

    print("DONE.")
    print("pT1: \(pT1 as AnyObject), pT2: \(pT2 as AnyObject)")

    XCTAssertEqual(pT1.rot.theta, pT2.rot.theta, accuracy: 1e-5)
  }

  /// TODO(fan): Change this to a proper noise model
  @differentiable
  func e_pose2(_ ŷ: Pose2) -> Double {
    // Squared error with Gaussian variance as weights
    0.1 * ŷ.rot.theta * ŷ.rot.theta + 0.3 * ŷ.t.x * ŷ.t.x + 0.3 * ŷ.t.y * ŷ.t.y
  }

  /// test convergence for a simple Pose2SLAM
  func testPose2SLAM() {
    let pi = 3.1415926

    let dumpjson = { (p: Pose2) -> String in
      "[ \(p.t.x), \(p.t.y), \(p.rot.theta)]"
    }

    // Initial estimate for poses
    let p1T0 = Pose2(Rot2(0.2), Vector2(0.5, 0.0))
    let p2T0 = Pose2(Rot2(-0.2), Vector2(2.3, 0.1))
    let p3T0 = Pose2(Rot2(pi / 2), Vector2(4.1, 0.1))
    let p4T0 = Pose2(Rot2(pi), Vector2(4.0, 2.0))
    let p5T0 = Pose2(Rot2(-pi / 2), Vector2(2.1, 2.1))

    var map = [p1T0, p2T0, p3T0, p4T0, p5T0]

    // print("map_history = [")
    for _ in 0..<1500 {
      let (_, 𝛁loss) = valueWithGradient(at: map) { map -> Double in
        var loss: Double = 0

        // Odometry measurements
        let p2T1 = between(between(map[1], map[0]), Pose2(2.0, 0.0, 0.0))
        let p3T2 = between(between(map[2], map[1]), Pose2(2.0, 0.0, pi / 2))
        let p4T3 = between(between(map[3], map[2]), Pose2(2.0, 0.0, pi / 2))
        let p5T4 = between(between(map[4], map[3]), Pose2(2.0, 0.0, pi / 2))

        // Sum through the errors
        let error = self.e_pose2(p2T1) + self.e_pose2(p3T2) + self.e_pose2(p4T3) + self.e_pose2(p5T4)
        loss = loss + (error / 3)

        return loss
      }

      // print("[")
      // for v in map.indices {
      //   print("\(dumpjson(map[v]))\({ () -> String in if v == map.indices.endIndex - 1 { return "" } else { return "," } }())")
      // }
      // print("],")

      // print("𝛁loss", 𝛁loss)
      // NOTE: this is more like sparse rep not matrix Jacobian
      map.move(along: 𝛁loss.scaled(by: -1.0))
    }

    // print("]")

    print("map = [")
    for v in map.indices {
      print("\(dumpjson(map[v]))\({ () -> String in if v == map.indices.endIndex - 1 { return "" } else { return "," } }())")
    }
    print("]")

    let p5T1 = between(map[4], map[0])

    // Test condition: P_5 should be identical to P_1 (close loop)
    XCTAssertEqual(p5T1.t.norm, 0.0, accuracy: 1e-2)
  }

  /// Tests that the derivative of the identity function is correct at a few random points.
  func testDerivativeIdentity() {
    func identity(_ x: Pose2) -> Pose2 {
      Pose2(x.rot, x.t)
    }
    for _ in 0..<10 {
      let expected: Tensor<Double> = eye(rowCount: 3)
      assertEqual(
        Tensor<Double>(
          matrixRows: jacobian(of: identity, at: Pose2(randomWithCovariance: eye(rowCount: 3)))),
        expected,
        accuracy: 1e-10
      )
    }
  }

  /// Test that the derivative of the group inverse operation is correct at a few random points.
  func testDerivativeInverse() {
    for _ in 0..<10 {
      let pose = Pose2(randomWithCovariance: eye(rowCount: 3))
      let expected = -pose.groupAdjointMatrix
      assertEqual(
        Tensor<Double>(matrixRows: jacobian(of: SwiftFusion.inverse, at: pose)),
        expected,
        accuracy: 1e-10
      )
    }
  }

  /// Test the the derivative of the group operations is correct at a few random points.
  func testDerivativeMultiplication() {
    func multiply(_ x: [Pose2]) -> Pose2 {
      x[0] * x[1]
    }
    for _ in 0..<10 {
      let lhs = Pose2(randomWithCovariance: eye(rowCount: 3))
      let rhs = Pose2(randomWithCovariance: eye(rowCount: 3))
      let expected = Tensor(
        concatenating: [SwiftFusion.inverse(rhs).groupAdjointMatrix, eye(rowCount: 3)],
        alongAxis: 1
      )
      assertEqual(
        Tensor<Double>(matrixRows: jacobian(of: multiply, at: [lhs, rhs])),
        expected,
        accuracy: 1e-10
      )
    }
  }

  /// tests a simple identity Jacobian for Pose2
  func testJacobianPose2Identity() {
    let wT1 = Pose2(1, 0, 3.1415926 / 2.0), wT2 = Pose2(1, 0, 3.1415926 / 2.0)
    let pts: [Pose2] = [wT1, wT2]

    let f: @differentiable(_ pts: [Pose2]) -> Double = { (_ pts: [Pose2]) -> Double in
      let d = between(pts[0], pts[1])

      return d.rot.theta * d.rot.theta + d.t.x * d.t.x + d.t.y * d.t.y
    }

    let j = jacobian(of: f, at: pts)
    // print("J(f) = \(j[0].base as AnyObject)")
    for item in j[0] {
      XCTAssertEqual(item, Pose2.TangentVector.zero)
    }
  }

  /// Tests the jacobian of the `between` function.
  func testJacobianPose2Trivial() {
    // Values taken from GTSAM `testPose2.cpp`
    let wT1 = Pose2(1, 2, .pi/2.0), wT2 = Pose2(-1, 4, .pi)
    let pts: [Pose2] = [wT1, wT2]

    let f: @differentiable(_ pts: [Pose2]) -> Pose2 = { (_ pts: [Pose2]) -> Pose2 in
      let d = between(pts[0], pts[1])

      return d
    }

    let j = jacobian(of: f, at: pts)

    // Note that these numbers are a permutation of the corresponding numbers from GTSAM because
    // the SwiftFusion convention for tangent vector is (omega, v) while the GTSAM convention is
    // (v, omega).
    let expected = Tensor<Double>([
      [-1.0, 0.0,  0.0, 1.0, 0.0, 0.0],
      [-2.0, 0.0, -1.0, 0.0, 1.0, 0.0],
      [-2.0, 1.0,  0.0, 0.0, 0.0, 1.0]
    ])

    assertEqual(Tensor<Double>(matrixRows: j), expected, accuracy: 1e-10)
  }

  /// test convergence for a simple Pose2SLAM
  func testPose2SLAMWithSGD() {
    let pi = 3.1415926

    let dumpjson = { (p: Pose2) -> String in
      "[ \(p.t.x), \(p.t.y), \(p.rot.theta)]"
    }

    // Initial estimate for poses
    let p1T0 = Pose2(Rot2(0.2), Vector2(0.5, 0.0))
    let p2T0 = Pose2(Rot2(-0.2), Vector2(2.3, 0.1))
    let p3T0 = Pose2(Rot2(pi / 2), Vector2(4.1, 0.1))
    let p4T0 = Pose2(Rot2(pi), Vector2(4.0, 2.0))
    let p5T0 = Pose2(Rot2(-pi / 2), Vector2(2.1, 2.1))

    var map = [p1T0, p2T0, p3T0, p4T0, p5T0]

    let optimizer = SGD(for: map, learningRate: 1.2)

    // print("map_history = [")
    for _ in 0..<500 {
      let (_, 𝛁loss) = valueWithGradient(at: map) { map -> Double in
        var loss: Double = 0

        // Odometry measurements
        let p2T1 = between(between(map[1], map[0]), Pose2(2.0, 0.0, 0.0))
        let p3T2 = between(between(map[2], map[1]), Pose2(2.0, 0.0, pi / 2))
        let p4T3 = between(between(map[3], map[2]), Pose2(2.0, 0.0, pi / 2))
        let p5T4 = between(between(map[4], map[3]), Pose2(2.0, 0.0, pi / 2))

        // Sum through the errors
        let error = self.e_pose2(p2T1) + self.e_pose2(p3T2) + self.e_pose2(p4T3) + self.e_pose2(p5T4)
        loss = loss + (error / 3)

        return loss
      }

      // print("[")
      // for v in map.indices {
      //   print("\(dumpjson(map[v]))\({ () -> String in if v == map.indices.endIndex - 1 { return "" } else { return "," } }())")
      // }
      // print("],")

      // print("𝛁loss", 𝛁loss)
      // NOTE: this is more like sparse rep not matrix Jacobian
      optimizer.update(&map, along: 𝛁loss)
    }

    // print("]")

    print("map = [")
    for v in map.indices {
      print("\(dumpjson(map[v]))\({ () -> String in if v == map.indices.endIndex - 1 { return "" } else { return "," } }())")
    }
    print("]")

    let p5T1 = between(map[4], map[0])

    // Test condition: P_5 should be identical to P_1 (close loop)
    XCTAssertEqual(p5T1.t.norm, 0.0, accuracy: 1e-2)
  }

  static var allTests = [
    ("testBetweenIdentitiesTrivial", testBetweenIdentitiesTrivial),
    ("testBetweenIdentities", testBetweenIdentities),
    ("testBetweenIdentities", testBetweenIdentitiesRotated),
    ("testBetweenDerivatives", testBetweenDerivatives),
    ("testDerivativeIdentity", testDerivativeIdentity),
    ("testDerivativeInverse", testDerivativeInverse),
    ("testDerivativeMultiplication", testDerivativeMultiplication),
    ("testJacobianPose2Identity", testJacobianPose2Identity),
    ("testJacobianPose2Trivial", testJacobianPose2Trivial),
    ("testPose2SLAM", testPose2SLAM),
  ]
}
