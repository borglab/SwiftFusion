import SwiftFusion
import XCTest

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
    var pT1 = Pose2(Rot2(0), Point2(1, 0)), pT2 = Pose2(Rot2(1), Point2(1, 1))

    for _ in 0..<100 {
      var (_, 𝛁loss) = valueWithGradient(at: pT1) { pT1 -> Double in
        var loss: Double = 0
        let ŷ = between(pT1, pT2)
        let error = ŷ.rot_.theta * ŷ.rot_.theta + ŷ.t_.x * ŷ.t_.x + ŷ.t_.y * ŷ.t_.y
        loss = loss + (error / 10)

        return loss
      }

      // print("𝛁loss", 𝛁loss)
      𝛁loss.rot_ = -𝛁loss.rot_
      𝛁loss.t_.x = -𝛁loss.t_.x
      𝛁loss.t_.y = -𝛁loss.t_.y
      pT1.move(along: 𝛁loss)
    }

    print("DONE.")
    print("pT1: \(pT1 as AnyObject), pT2: \(pT2 as AnyObject)")

    XCTAssertEqual(pT1.rot_.theta, pT2.rot_.theta, accuracy: 1e-5)
  }

  /// TODO(fan): Change this to a proper noise model
  @differentiable
  func e_pose2(_ ŷ: Pose2) -> Double {
    // Squared error with Gaussian variance as weights
    0.1 * ŷ.rot_.theta * ŷ.rot_.theta + 0.3 * ŷ.t_.x * ŷ.t_.x + 0.3 * ŷ.t_.y * ŷ.t_.y
  }

  /// test convergence for a simple Pose2SLAM
  func testPose2SLAM() {
    let pi = 3.1415926

    let dumpjson = { (p: Pose2) -> String in
      "[ \(p.t_.x), \(p.t_.y), \(p.rot_.theta)]"
    }

    // Initial estimate for poses
    let p1T0 = Pose2(Rot2(0.2), Point2(0.5, 0.0))
    let p2T0 = Pose2(Rot2(-0.2), Point2(2.3, 0.1))
    let p3T0 = Pose2(Rot2(pi / 2), Point2(4.1, 0.1))
    let p4T0 = Pose2(Rot2(pi), Point2(4.0, 2.0))
    let p5T0 = Pose2(Rot2(-pi / 2), Point2(2.1, 2.1))

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
    XCTAssertEqual(p5T1.t_.magnitude, 0.0, accuracy: 1e-2)
  }

  /// test convergence for a simple Pose2SLAM
  func testPose2SLAMWithSGD() {
    let pi = 3.1415926

    let dumpjson = { (p: Pose2) -> String in
      "[ \(p.t_.x), \(p.t_.y), \(p.rot_.theta)]"
    }

    // Initial estimate for poses
    let p1T0 = Pose2(Rot2(0.2), Point2(0.5, 0.0))
    let p2T0 = Pose2(Rot2(-0.2), Point2(2.3, 0.1))
    let p3T0 = Pose2(Rot2(pi / 2), Point2(4.1, 0.1))
    let p4T0 = Pose2(Rot2(pi), Point2(4.0, 2.0))
    let p5T0 = Pose2(Rot2(-pi / 2), Point2(2.1, 2.1))

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
    XCTAssertEqual(p5T1.t_.magnitude, 0.0, accuracy: 1e-2)
  }

  static var allTests = [
    ("testBetweenIdentitiesTrivial", testBetweenIdentitiesTrivial),
    ("testBetweenIdentities", testBetweenIdentities),
    ("testBetweenIdentities", testBetweenIdentitiesRotated),
    ("testBetweenDerivatives", testBetweenDerivatives),
    ("testPose2SLAM", testPose2SLAM),
  ]
}
