import XCTest
import SwiftFusion
import TensorFlow

final class Pose3Tests: XCTestCase {
  /// Tests that the manifold invariant holds for Pose2
  func testManifoldIdentity() {
    for _ in 0..<10 {
      let p = Pose3.fromTangent(Vector6(Tensor<Double>(randomNormal: [6])))
      let q = Pose3.fromTangent(Vector6(Tensor<Double>(randomNormal: [6])))
      let actual: Pose3 = Pose3(coordinate: p.coordinate.retract(p.coordinate.localCoordinate(q.coordinate)))
      assertAllKeyPathEqual(actual, q, accuracy: 1e-10)
    }
  }
  
  func testTrivialRotation() {
    let p = Pose3.fromTangent(Vector6(Tensor<Double>([0.3, 0, 0, 0, 0, 0])))
    let R = Rot3.fromTangent(Vector3(0.3, 0, 0))
    
    assertAllKeyPathEqual(p.rot, R, accuracy: 1e-10)
  }
  
  func testTrivialFull1() {
    let P = Vector3(0.2,0.7,-2)
    let R = Rot3.fromTangent(Vector3(0.3, 0, 0))
    let p = Pose3.fromTangent(Vector6(Tensor<Double>([0.3, 0, 0, 0.2, 0.394742, -2.08998])))
    assertAllKeyPathEqual(p.rot, R, accuracy: 1e-10)
    assertAllKeyPathEqual(p.t, P, accuracy: 1e-5)
  }
  
  func testTrivialFull2() {
    let P = Vector3(3.5,-8.2,4.2)
    let R = Rot3.fromTangent(Vector3(0.3, 0, 0))
    let t12 = Vector6(Tensor<Double>(repeating: 0.1, shape: [6]))
    let t1 = Pose3(R, P)
    let t2 = Pose3(coordinate: t1.coordinate.retract(t12))
    assertAllKeyPathEqual(t1.coordinate.localCoordinate(t2.coordinate), t12, accuracy: 1e-5)
  }
  
  func testPose3SimplePriorFactor() {
    let P = Vector3(3.5,-8.2,4.2)
    let R = Rot3.fromTangent(Vector3(0.3, 0, 0))
    let t1 = Pose3(R, P)
    let I: Tensor<Double> = eye(rowCount: 6)
    let prior_factor = PriorFactor(0, t1)
    
    var vals = Values()
    vals.insert(0, t1) // should be identity matrix
    // Change this to t2, still zero in upper left block
    
    let actual = prior_factor.linearize(vals).jacobians[0]
    
    assertEqual(actual.tensor, I, accuracy: 1e-8)
  }
  
  /// circlePose3 generates a set of poses in a circle. This function
  /// returns those poses inside a gtsam.Values object, with sequential
  /// keys starting from 0. An optional character may be provided, which
  /// will be stored in the msb of each key (i.e. gtsam.Symbol).

  /// We use aerospace/navlab convention, X forward, Y right, Z down
  /// First pose will be at (R,0,0)
  /// ^y   ^ X
  /// |    |
  /// z-->xZ--> Y  (z pointing towards viewer, Z pointing away from viewer)
  /// Vehicle at p0 is looking towards y axis (X-axis points towards world y)
  func circlePose3(numPoses: Int = 8, radius: Double = 1.0) -> Values {
    var values = Values()
    var theta = 0.0
    let dtheta = 2.0 * .pi / Double(numPoses)
    let gRo = Rot3(0, 1, 0, 1, 0, 0, 0, 0, -1)
    for i in 0..<numPoses {
      let key = 0 + i
      let gti = Vector3(radius * cos(theta), radius * sin(theta), 0)
      let oRi = Rot3.fromTangent(Vector3(0, 0, -theta))  // negative yaw goes counterclockwise, with Z down !
      let gTi = Pose3(gRo * oRi, gti)
      values.insert(key, gTi)
      theta = theta + dtheta
    }
    return values
  }
  
  func testGtsamPose3SLAMExample() {
    // Create a hexagon of poses
    let hexagon = circlePose3(numPoses: 6, radius: 1.0)
    let p0 = hexagon[0, as: Pose3.self]
    let p1 = hexagon[1, as: Pose3.self]
    
    // create a Pose graph with one equality constraint and one measurement
    var fg = NonlinearFactorGraph()
    fg += PriorFactor(0, p0)
    let delta = between(p0, p1)

    fg += BetweenFactor(0, 1, delta)
    fg += BetweenFactor(1, 2, delta)
    fg += BetweenFactor(2, 3, delta)
    fg += BetweenFactor(3, 4, delta)
    fg += BetweenFactor(4, 5, delta)
    fg += BetweenFactor(5, 0, delta)

    // Create initial config
    var val = Values()
    let s = 0.10
    val.insert(0, p0)
    val.insert(1, hexagon[1, as: Pose3.self].retract(Vector6(s * Tensor<Double>(randomNormal: [6]))))
    val.insert(2, hexagon[2, as: Pose3.self].retract(Vector6(s * Tensor<Double>(randomNormal: [6]))))
    val.insert(3, hexagon[3, as: Pose3.self].retract(Vector6(s * Tensor<Double>(randomNormal: [6]))))
    val.insert(4, hexagon[4, as: Pose3.self].retract(Vector6(s * Tensor<Double>(randomNormal: [6]))))
    val.insert(5, hexagon[5, as: Pose3.self].retract(Vector6(s * Tensor<Double>(randomNormal: [6]))))

    // optimize
    for _ in 0..<16 {
      let gfg = fg.linearize(val)
      
      let optimizer = CGLS(precision: 1e-6, max_iteration: 500)
      
      var dx = VectorValues()
      
      for i in 0..<6 {
        dx.insert(i, Vector(zeros: 6))
      }
      
      optimizer.optimize(objective: gfg, initial: &dx)
      
      val.move(along: dx)
    }

    let pose_1 = val[1, as: Pose3.self]
    assertAllKeyPathEqual(pose_1, p1, accuracy: 1e-2)
  }
}
