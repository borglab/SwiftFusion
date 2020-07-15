import XCTest
import SwiftFusion
import PenguinStructures
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
    let I = FixedSizeMatrix<Array6<Tuple1<Vector6>>>.identity
    
    var val = VariableAssignments()
    let id1 = val.store(t1) // should be identity matrix
    let prior_factor = PriorFactor(id1, t1)
    // Change this to t2, still zero in upper left block
    
    let actual = JacobianFactor6x6_1(linearizing: prior_factor, at: Tuple1(t1))
    
    let jacobian = actual.jacobian

    // print(voidPtr.pointee)
    XCTAssertEqual(jacobian, I.rows)
  }
  
  /// circlePose3 generates a set of poses in a circle. This function
  /// returns those poses inside a gtsam.VariableAssignments object, with sequential
  /// keys starting from 0. An optional character may be provided, which
  /// will be stored in the msb of each key (i.e. gtsam.Symbol).

  /// We use aerospace/navlab convention, X forward, Y right, Z down
  /// First pose will be at (R,0,0)
  /// ^y   ^ X
  /// |    |
  /// z-->xZ--> Y  (z pointing towards viewer, Z pointing away from viewer)
  /// Vehicle at p0 is looking towards y axis (X-axis points towards world y)
  func circlePose3(numPoses: Int = 8, radius: Double = 1.0) -> (Array<TypedID<Pose3>>, VariableAssignments) {
    var values = VariableAssignments()
    var ids = [TypedID<Pose3>]()
    var theta: Double = 0.0
    let dtheta = 2.0 * .pi / Double(numPoses)
    let gRo = Rot3(0, 1, 0, 1, 0, 0, 0, 0, -1)
    for _ in 0..<numPoses {
      let gti = Vector3(radius * cos(theta), radius * sin(theta), 0)
      let oRi = Rot3.fromTangent(Vector3(0, 0, -theta))  // negative yaw goes counterclockwise, with Z down !
      let gTi = Pose3(gRo * oRi, gti)
      ids.append(values.store(gTi))
      theta = theta + dtheta
    }
    return (ids, values)
  }
  
  func testGtsamPose3SLAMExample() {
    // Create a hexagon of poses
    let (hexagon_id, hexagon_val) = circlePose3(numPoses: 6, radius: 1.0)
    let p0 = hexagon_val[hexagon_id[0]]
    let p1 = hexagon_val[hexagon_id[1]]
    
    // create a Pose graph with one equality constraint and one measurement
    var fg = FactorGraph()
    fg.store(PriorFactor(hexagon_id[0], p0))
    let delta = between(p0, p1)

    fg.store(BetweenFactor(hexagon_id[0], hexagon_id[1], delta))
    fg.store(BetweenFactor(hexagon_id[1], hexagon_id[2], delta))
    fg.store(BetweenFactor(hexagon_id[2], hexagon_id[3], delta))
    fg.store(BetweenFactor(hexagon_id[3], hexagon_id[4], delta))
    fg.store(BetweenFactor(hexagon_id[4], hexagon_id[5], delta))
    fg.store(BetweenFactor(hexagon_id[5], hexagon_id[0], delta))

    // Create initial config
    var val = VariableAssignments()
    let s: Double = 0.10
    let _ = val.store(p0)
    let _ = val.store(hexagon_val[hexagon_id[1]].retract(Vector6(s * Tensor<Double>(randomNormal: [6]))))
    let _ = val.store(hexagon_val[hexagon_id[2]].retract(Vector6(s * Tensor<Double>(randomNormal: [6]))))
    let _ = val.store(hexagon_val[hexagon_id[3]].retract(Vector6(s * Tensor<Double>(randomNormal: [6]))))
    let _ = val.store(hexagon_val[hexagon_id[4]].retract(Vector6(s * Tensor<Double>(randomNormal: [6]))))
    let _ = val.store(hexagon_val[hexagon_id[5]].retract(Vector6(s * Tensor<Double>(randomNormal: [6]))))

    // optimize
    for _ in 0..<16 {
      let gfg = fg.linearized(at: val)
      
      var optimizer = GenericCGLS(precision: 1e-6, max_iteration: 500)
      
      var dx = val.tangentVectorZeros

      optimizer.optimize(gfg: gfg, initial: &dx)
      
      val.move(along: dx)
    }

    let pose_1 = val[hexagon_id[1]]
    assertAllKeyPathEqual(pose_1, p1, accuracy: 1e-2)
  }

  /// Tests that the custom implementation of `Adjoint` is correct.
  func testAdjoint() {
    for _ in 0..<10 {
      let pose = Pose3.fromTangent(Vector6(Tensor<Double>(randomNormal: [6])))
      for v in Pose3.TangentVector.standardBasis {
        assertEqual(
          pose.Adjoint(v).tensor,
          pose.coordinate.defaultAdjoint(v).tensor,
          accuracy: 1e-10
        )
      }
    }
  }

  /// Tests that the custom implementation of `AdjointTranspose` is correct.
  func testAdjointTranspose() {
    for _ in 0..<10 {
      let pose = Pose3.fromTangent(Vector6(Tensor<Double>(randomNormal: [6])))
      for v in Pose3.TangentVector.standardBasis {
        assertEqual(
          pose.AdjointTranspose(v).tensor,
          pose.coordinate.defaultAdjointTranspose(v).tensor,
          accuracy: 1e-10
        )
      }
    }
  }
}
