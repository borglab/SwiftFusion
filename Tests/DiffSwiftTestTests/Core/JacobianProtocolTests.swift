import Foundation
import XCTest

@testable import DiffSwiftTest

class JacobianProtocolTests: XCTestCase {
  static var allTests = [
    ("testJacobianPose2Identity", testJacobianPose2Identity),
    ("testJacobianPose2Trivial", testJacobianPose2Trivial),
  ]

  // --------------------------------------------------------------------------
  // testConcat
  func testJacobianPose2Identity() {
    let wT1 = Pose2(1, 0, 3.1415926 / 2.0), wT2 = Pose2(1, 0, 3.1415926 / 2.0)
    let map: [Pose2] = [wT1, wT2]

    let ef: @differentiable(_ map: [Pose2]) -> Double = { (_ map: [Pose2]) -> Double in
      let d = between(map[0], map[1])

      return d.rot_.theta * d.rot_.theta + d.t_.x * d.t_.x + d.t_.y * d.t_.y
    }

    let j = jacobian(of: ef, at: map)
    print("J(ef) = \(j[0].base as AnyObject)")
    for item in j[0] {
      XCTAssertEqual(item, Pose2.TangentVector.zero)
    }
  }

  func testJacobianPose2Trivial() {
    let wT1 = Pose2(1, 0, 3.1415926 / 2.0), wT2 = Pose2(1, 1, -3.1415926 / 2.0)
    let map: [Pose2] = [wT1, wT2]

    let ef: @differentiable(_ map: [Pose2]) -> Double = { (_ map: [Pose2]) -> Double in
      let d = between(map[0], map[1])

      return d.rot_.theta * d.rot_.theta + d.t_.x * d.t_.x + d.t_.y * d.t_.y
    }

    let j = jacobian(of: ef, at: map)
    
    print("j(ef) = \(j[0].base as AnyObject)")

    print("Pose2.TangentVector.basisVectors() = \(Pose2.TangentVector.basisVectors() as AnyObject)")

    print("J(ef) = [")
    for r in j[0] {
      print(r.recursivelyAllKeyPaths(to:Double.self).map {r[keyPath: $0]})
    }
    print("]")
  }
    
  func testJacobian2D() {
    let p1 = Point2(0, 1), p2 = Point2(0,0), p3 = Point2(0,0);
    let map: [Point2] = [p1, p2, p3]

    let ef: @differentiable(_ map: [Point2]) -> Point2 = { (_ map: [Point2]) -> Point2 in
      let d = map[1] - map[0]

      return d
    }

    let j = jacobian(of: ef, at: map)
    
    print("j(ef) = \(j as AnyObject)")

    print("Point2.TangentVector.basisVectors() = \(Point2.TangentVector.basisVectors() as AnyObject)")
    
    /* Example output:
      J(ef) = [
      [ [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.0] ]
      [ [0.0, -1.0],
        [0.0, 1.0],
        [0.0, 0.0] ]
      ]
     So this is 2x3 but the data type is Point2.TangentVector.
     In "normal" Jacobian notation, we should have a 2x6.
     [ [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
       [0.0, -1.0, 0.0, 1.0, 0.0, 0.0] ]
    */
    print("J(ef) = [")
    for c in j {
      print("[")
      for r in c {
        print(r.recursivelyAllKeyPaths(to:Double.self).map {r[keyPath: $0]})
        print(",")
      }
      print("]")
    }
    print("]")
  }
}
