// This file tests for the CGLS optimizer

import SwiftFusion
import TensorFlow
import XCTest

final class LMTests: XCTestCase {
  /// test convergence for a simple gaussian factor graph
  func testBasicLMConvergence() {
    var x = VariableAssignments()
    let pose1ID = x.store(Pose2(Rot2(0.2), Vector2(0.5, 0.0)))
    let pose2ID = x.store(Pose2(Rot2(-0.2), Vector2(2.3, 0.1)))
    let pose3ID = x.store(Pose2(Rot2(.pi / 2), Vector2(4.1, 0.1)))
    let pose4ID = x.store(Pose2(Rot2(.pi), Vector2(4.0, 2.0)))
    let pose5ID = x.store(Pose2(Rot2(-.pi / 2), Vector2(2.1, 2.1)))

    var graph = FactorGraph()
    graph.store(BetweenFactor2(pose2ID, pose1ID, Pose2(2.0, 0.0, .pi / 2)))
    graph.store(BetweenFactor2(pose3ID, pose2ID, Pose2(2.0, 0.0, .pi / 2)))
    graph.store(BetweenFactor2(pose4ID, pose3ID, Pose2(2.0, 0.0, .pi / 2)))
    graph.store(BetweenFactor2(pose5ID, pose4ID, Pose2(2.0, 0.0, .pi / 2)))
    graph.store(PriorFactor2(pose1ID, Pose2(0, 0, 0)))

    var optimizer = LM(precision: 1e-3, max_iteration: 10)
    optimizer.verbosity = .TRYLAMBDA
    
    try? optimizer.optimize(graph: graph, initial: &x)

    // Test condition: pose 5 should be identical to pose 1 (close loop).
    XCTAssertEqual(between(x[pose1ID], x[pose5ID]).t.norm, 0.0, accuracy: 1e-2)
  }
}
