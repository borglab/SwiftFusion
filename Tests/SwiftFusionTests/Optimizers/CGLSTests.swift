// This file tests for the CGLS optimizer

import SwiftFusion
import XCTest

final class CGLSTests: XCTestCase {
  /// TODO(fan): Change this to a proper noise model
  @differentiable
  func e_pose2(_ ŷ: Pose2) -> Double {
    // Squared error with Gaussian variance as weights
    0.1 * ŷ.rot_.theta * ŷ.rot_.theta + 0.3 * ŷ.t_.x * ŷ.t_.x + 0.3 * ŷ.t_.y * ŷ.t_.y
  }

  /// test convergence for a simple Pose2SLAM
  func testPose2SLAMWithCGLS() {
    let pi = 3.1415926

    let dumpjson = { (p: Pose2) -> String in
      "[ \(p.t_.x), \(p.t_.y), \(p.rot_.theta)]"
    }

    // Initial estimate for poses
    let p1T0 = Pose2(Rot2(0.2), Vector2(0.5, 0.0))
    let p2T0 = Pose2(Rot2(-0.2), Vector2(2.3, 0.1))
    let p3T0 = Pose2(Rot2(pi / 2), Vector2(4.1, 0.1))
    let p4T0 = Pose2(Rot2(pi), Vector2(4.0, 2.0))
    let p5T0 = Pose2(Rot2(-pi / 2), Vector2(2.1, 2.1))

    var graph = NonlinearFactorGraph()
    var poses = Values()
    
    // Odometry measurements
    let p2T1 = BetweenFactor(1, 0, Pose2(2.0, 0.0, 0.0))
    let p3T2 = BetweenFactor(2, 1 Pose2(2.0, 0.0, pi / 2))
    let p4T3 = BetweenFactor(3, 2, Pose2(2.0, 0.0, pi / 2))
    let p5T4 = BetweenFactor(4, 3, Pose2(2.0, 0.0, pi / 2))
    
    let loop_constraint =
      BetweenFactor(4, 0, Pose2(0.0, 0.0, pi / 2), NoiseModel.Gaussian(0.001))
    
    graph.insert(p2T1)
    graph.insert(p3T2)
    graph.insert(p4T3)
    graph.insert(p5T4)
    graph.insert(loop_constraint)
    
    poses.insert(0, p1T0)
    poses.insert(1, p2T0)
    poses.insert(2, p3T0)
    poses.insert(3, p4T0)
    poses.insert(4, p5T0)
    
    for _ in 0..<500 {
      let gfg = graph.linearize(at: poses)
      
      let optimizer = CGLS()
      
      let new_pose = optimizer.optimize(gfg: gfg, initial: poses)
      
      // poses.move(tangent.scaled(-1.0))
      pose = new_pose
    }

    let p5T1 = between(poses[4], poses[0])

    // Test condition: P_5 should be identical to P_1 (close loop)
    XCTAssertEqual(p5T1.t.norm, 0.0, accuracy: 1e-2)
  }

  static var allTests = [
    ("testPose2SLAMWithSGD", testPose2SLAMWithCGLS),
  ]
}
