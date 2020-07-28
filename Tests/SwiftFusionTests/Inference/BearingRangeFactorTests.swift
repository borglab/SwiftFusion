import SwiftFusion
import XCTest

/// Tests that `BearingRangeError` satisfies the `EuclideanVector` requirements.
final class BearingRangeFactorTests: XCTestCase {
  func testPlanarSLAM() {
    var fg = FactorGraph()
    var val = VariableAssignments()
    let X1 = val.store(Pose2(-0.25, 0.20, 0.15))
    let X2 = val.store(Pose2(2.30, 0.10, -0.20))
    let X3 = val.store(Pose2(4.10, 0.10, 0.10))
    let L1 = val.store(Vector2(1.80, 2.10))
    let L2 = val.store(Vector2(4.10, 1.80))


    // Add a prior on pose X1 at the origin. A prior factor consists of a mean and a noise model
    fg.store(PriorFactor(X1, Pose2(0.0, 0.0, 0.0)))

    // Add odometry factors between X1,X2 and X2,X3, respectively
    fg.store(BetweenFactor(
        X1, X2, Pose2(2.0, 0.0, 0.0)))
    fg.store(BetweenFactor(
        X2, X3, Pose2(2.0, 0.0, 0.0)))

    // Add Range-Bearing measurements to two different landmarks L1 and L2
    fg.store(BearingRangeFactor2(
      X1, L1, Rot2(45 / 180 * .pi), sqrt(4.0+4.0)))
    fg.store(BearingRangeFactor2(
      X2, L1, Rot2(90 / 180 * .pi), 2.0))
    fg.store(BearingRangeFactor2(
      X3, L2, Rot2(90 / 180 * .pi), 2.0))

    // Create (deliberately inaccurate) initial estimate
    
    
    for _ in 0..<10 {
      let gfg = fg.linearized(at: val)
      
      var optimizer = GenericCGLS(precision: 1e-6, max_iteration: 500)
      
      var dx = val.tangentVectorZeros
      
      optimizer.optimize(gfg: gfg, initial: &dx)
      
      val.move(along: dx)
    }
    
    // Test condition: P_5 should be identical to P_1 (close loop)
    XCTAssertEqual(fg.error(at: val), 0.0, accuracy: 1e-3)
  }
}
