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
    assertEqual(gfg.residual(vv).values[0].tensor, expected, accuracy: 1e-6)
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
      
      optimizer.optimize(gfg, initial: &dx)
      
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
}
