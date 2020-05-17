import SwiftFusion
import TensorFlow
import XCTest

final class NonlinearFactorGraphTests: XCTestCase {
  /// test ATr
  func testBasicOps() {
    var fg = NonlinearFactorGraph()
    
    let bf1 = BetweenFactor(0, 1, Pose2(0.0,0.0, 0.0))
    
    fg += bf1
    
    var val = Values()
    val.insert(0, Pose2(1.0, 1.0, 0.0))
    val.insert(1, Pose2(1.0, 1.0, .pi))
    
    let gfg = fg.linearize(val)
    
    var vv = VectorValues()
    
    vv.insert(0, Vector(zeros: 3))
    vv.insert(1, Vector(zeros: 3))
    
    let expected = Tensor<Double>(shape:[3], scalars: [.pi, 0.0, 0.0])
    
    print("gfg = \(gfg)")
    print("error = \(gfg.residual(vv).norm)")
    assertEqual((gfg.residual(vv))[0].tensor, expected, accuracy: 1e-6)
  }
  
  /// test CGLS iterative solver
  func testCGLSPose2SLAM() {
    // Initial estimate for poses
    let p1T0 = Pose2(Rot2(0.2), Vector2(0.5, 0.0))
    let p2T0 = Pose2(Rot2(-0.2), Vector2(2.3, 0.1))
    let p3T0 = Pose2(Rot2(.pi / 2), Vector2(4.1, 0.1))
    let p4T0 = Pose2(Rot2(.pi), Vector2(4.0, 2.0))
    let p5T0 = Pose2(Rot2(-.pi / 2), Vector2(2.1, 2.1))

    let map = [p1T0, p2T0, p3T0, p4T0, p5T0]

    var fg = NonlinearFactorGraph()
    
    fg += BetweenFactor(1, 0, Pose2(2.0, 0.0, .pi / 2))
    fg += BetweenFactor(2, 1, Pose2(2.0, 0.0, .pi / 2))
    fg += BetweenFactor(3, 2, Pose2(2.0, 0.0, .pi / 2))
    fg += BetweenFactor(4, 3, Pose2(2.0, 0.0, .pi / 2))
    fg += PriorFactor(0, Pose2(0.0, 0.0, 0.0))
    
    var val = Values()
    
    for i in 0..<5 {
      val.insert(i, map[i])
    }
    
    for _ in 0..<3 {
      let gfg = fg.linearize(val)
      
      let optimizer = CGLS(precision: 1e-6, max_iteration: 500)
      
      var dx = VectorValues()
      
      for i in 0..<5 {
        dx.insert(i, Vector(zeros: 3))
      }
      
      optimizer.optimize(gfg: gfg, initial: &dx)
      
      val.move(along: dx)
    }
    
    let dumpjson = { (p: Pose2) -> String in
      "[ \(p.rot.theta), \(p.t.x), \(p.t.y)]"
    }
    
    print("map_init = [")
    for v in map.indices {
      print("\(dumpjson(map[v]))\({ () -> String in if v == map.indices.endIndex - 1 { return "" } else { return "," } }())")
    }
    print("]")
    
    let map_final = (0..<5).map { val[$0, as: Pose2.self] }
    print("map = [")
    for v in map_final.indices {
      print("\(dumpjson(map_final[v]))\({ () -> String in if v == map_final.indices.endIndex - 1 { return "" } else { return "," } }())")
    }
    print("]")
    
    let p5T1 = between(val[4, as: Pose2.self], val[0, as: Pose2.self])

    // Test condition: P_5 should be identical to P_1 (close loop)
    XCTAssertEqual(p5T1.t.norm, 0.0, accuracy: 1e-2)
  }
  
  func testPlanarSLAM() {
    var fg = NonlinearFactorGraph()
    
    let X1 = 0
    let X2 = 1
    let X3 = 2
    let L1 = 3
    let L2 = 4

    // Add a prior on pose X1 at the origin. A prior factor consists of a mean and a noise model
    fg += PriorFactor(X1, Pose2(0.0, 0.0, 0.0))

    // Add odometry factors between X1,X2 and X2,X3, respectively
    fg += BetweenFactor(
        X1, X2, Pose2(2.0, 0.0, 0.0))
    fg += BetweenFactor(
        X2, X3, Pose2(2.0, 0.0, 0.0))

    // Add Range-Bearing measurements to two different landmarks L1 and L2
    fg += BearingRangeFactor2D(
      X1, L1, 45 / 180 * .pi, sqrt(4.0+4.0))
    fg += BearingRangeFactor2D(
      X2, L1, 90 / 180 * .pi, 2.0)
    fg += BearingRangeFactor2D(
      X3, L2, 90 / 180 * .pi, 2.0)

    // Create (deliberately inaccurate) initial estimate
    var val = Values()
    val.insert(X1, Pose2(-0.25, 0.20, 0.15))
    val.insert(X2, Pose2(2.30, 0.10, -0.20))
    val.insert(X3, Pose2(4.10, 0.10, 0.10))
    val.insert(L1, Vector2(1.80, 2.10))
    val.insert(L2, Vector2(4.10, 1.80))
    
    for _ in 0..<10 {
      let gfg = fg.linearize(val)
      
      let optimizer = CGLS(precision: 1e-6, max_iteration: 500)
      
      var dx = VectorValues()
      
      for i in 0..<3 {
        dx.insert(i, Vector(zeros: 3))
      }
      
      for i in 3..<5 {
        dx.insert(i, Vector(zeros: 2))
      }
      
      optimizer.optimize(gfg: gfg, initial: &dx)
      
      val.move(along: dx)
      print("Current loss = \(fg.error(val))")
    }
    
    print("Final estimate:")
    print(val)
    // Test condition: P_5 should be identical to P_1 (close loop)
//    XCTAssertEqual(p5T1.t.norm, 0.0, accuracy: 1e-2)
  }
}
