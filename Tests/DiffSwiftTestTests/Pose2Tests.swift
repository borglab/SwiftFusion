@testable import DiffSwiftTest
import XCTest

final class Pose2Tests: XCTestCase {
  /// test between for trivial values
  func testBetweenIdentitiesTrivial() {
    let wT1 = Pose2(0,0,0), wT2 = Pose2(0,0,0);
    let expected = Pose2(0,0,0);
    let actual = between(wT1, wT2);
    XCTAssertEqual(actual, expected)
  }

  /// test between function for non-rotated poses
  func testBetweenIdentities() {
    let wT1 = Pose2(2,1,0), wT2 = Pose2(5, 2, 0);
    let expected = Pose2(3, 1,0);
    let actual = between(wT1, wT2);
    XCTAssertEqual(actual, expected)
  }

  /// test between function for rotated poses
  func testBetweenIdentitiesRotated() {
    let wT1 = Pose2(1,0,3.1415926/2.0), wT2 = Pose2(1, 0, -3.1415926/2.0);
    let expected = Pose2(0, 0,-3.1415926);
    let actual = between(wT1, wT2);
    //dump(expected, name: "expected");
    //dump(actual, name: "actual");
    XCTAssertEqual(actual, expected)
  }

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

  @differentiable
  func e_pose2 (_ ŷ: Pose2) -> Double {
    return 0.1 * ŷ.rot_.theta * ŷ.rot_.theta + 0.3 * ŷ.t_.x * ŷ.t_.x + 0.3 * ŷ.t_.y * ŷ.t_.y
  }

  func testPose2SLAM() {
    let pi = 3.1415926

    let p1T0 = Pose2(Rot2(0.2), Point2(0.5, 0.0))
    let p2T0 = Pose2(Rot2(-0.2), Point2(2.3, 0.1))
    let p3T0 = Pose2(Rot2(pi / 2), Point2(4.1, 0.1))
    let p4T0 = Pose2(Rot2(pi), Point2(4.0, 2.0))
    let p5T0 = Pose2(Rot2(-pi / 2), Point2(2.1, 2.1))

    var map = [p1T0, p2T0, p3T0, p4T0, p5T0]
    // graph.add(gtsam.BetweenFactorPose2(
    //     1, 2, gtsam.Pose2(2.0, 0.0, 0.0), odometryNoise))
    // graph.add(gtsam.BetweenFactorPose2(
    //     2, 3, gtsam.Pose2(2.0, 0.0, pi / 2), odometryNoise))
    // graph.add(gtsam.BetweenFactorPose2(
    //     3, 4, gtsam.Pose2(2.0, 0.0, pi / 2), odometryNoise))
    // graph.add(gtsam.BetweenFactorPose2(
    //     4, 5, gtsam.Pose2(2.0, 0.0, pi / 2), odometryNoise))

    for _ in 0..<500 {
      var (_, 𝛁loss) = valueWithGradient(at: map) { map -> Double in
        var loss: Double = 0
        let p2T1 = between(between(map[1], map[0]), Pose2(2.0, 0.0, 0.0))
        let p3T2 = between(between(map[2], map[1]), Pose2(2.0, 0.0, pi / 2))
        let p4T3 = between(between(map[3], map[2]), Pose2(2.0, 0.0, pi / 2))
        let p5T4 = between(between(map[4], map[3]), Pose2(2.0, 0.0, pi / 2))

        let error = self.e_pose2(p2T1) + self.e_pose2(p3T2) + self.e_pose2(p4T3) + self.e_pose2(p5T4)
        loss = loss - (error / 5)

        return loss
      }

      //print("𝛁loss", 𝛁loss)
      // 𝛁loss.rot_ = -𝛁loss.rot_
      // 𝛁loss.t_.x = -𝛁loss.t_.x
      // 𝛁loss.t_.y = -𝛁loss.t_.y
      map.move(along: 𝛁loss)
    }

    let dumpjson = { (p: Pose2) -> String in 
      return "[ \(p.t_.x), \(p.t_.y), \(p.rot_.theta)]"
    }

    print("DONE.")
    print("p1T0: \(dumpjson(map[0]))")
    print("p2T0: \(dumpjson(map[1]))")
    print("p3T0: \(dumpjson(map[2]))")
    print("p4T0: \(dumpjson(map[3]))")
    print("p5T0: \(dumpjson(map[4]))")
    // XCTAssertEqual(pT1.rot_.theta, pT2.rot_.theta, accuracy: 1e-5)
  }

  static var allTests = [
    ("testBetweenIdentitiesTrivial", testBetweenIdentitiesTrivial),
    ("testBetweenIdentities", testBetweenIdentities),
    ("testBetweenIdentities", testBetweenIdentitiesRotated),
    ("testBetweenDerivatives", testBetweenDerivatives),
    ("testPose2SLAM", testPose2SLAM)
  ]
}
