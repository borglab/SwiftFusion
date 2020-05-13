// This file tests for the NLCG optimizer

import SwiftFusion
import XCTest

final class NLCGTests: XCTestCase {
  /// TODO(fan): Change this to a proper noise model
  @differentiable
  func e_pose2(_ ŷ: Pose2) -> Double {
    // Squared error with Gaussian variance as weights
    0.1 * ŷ.rot.theta * ŷ.rot.theta + 0.3 * ŷ.t.x * ŷ.t.x + 0.3 * ŷ.t.y * ŷ.t.y
  }

  /// test convergence for a simple Pose2SLAM
  func testPose2SLAMWithNLCG() {
    let dumpjson = { (p: Pose2) -> String in
      "[ \(p.t.x), \(p.t.y), \(p.rot.theta)]"
    }

    // Initial estimate for poses
    let p1T0 = Pose2(Rot2(0.2), Vector2(0.5, 0.0))
    let p2T0 = Pose2(Rot2(-0.2), Vector2(2.3, 0.1))
    let p3T0 = Pose2(Rot2(.pi / 2), Vector2(4.1, 0.1))
    let p4T0 = Pose2(Rot2(.pi), Vector2(4.0, 2.0))
    let p5T0 = Pose2(Rot2(-.pi / 2), Vector2(2.1, 2.1))

    var map = [p1T0, p2T0, p3T0, p4T0, p5T0]

    let optimizer = NLCG(for: map, max_iteration: 170)
    
    let loss: @differentiable (_ map: Array<Pose2>) -> Double = { map -> Double in
      var loss: Double = 0

      // Odometry measurements
      let p2T1 = between(between(map[1], map[0]), Pose2(2.0, 0.0, 0.0))
      let p3T2 = between(between(map[2], map[1]), Pose2(2.0, 0.0, .pi / 2))
      let p4T3 = between(between(map[3], map[2]), Pose2(2.0, 0.0, .pi / 2))
      let p5T4 = between(between(map[4], map[3]), Pose2(2.0, 0.0, .pi / 2))

      // Sum through the errors
      let error = self.e_pose2(p2T1) + self.e_pose2(p3T2) + self.e_pose2(p4T3) + self.e_pose2(p5T4)
      loss = loss + (error / 3)

      return loss
    }
    
    optimizer.optimize(loss: loss, model: &map)
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
}
