import Foundation
import TensorFlow
import XCTest

import SwiftFusion

class JacobianTests: XCTestCase {
  static var allTests = [
    ("testJacobianPose2Identity", testJacobianPose2Identity),
    ("testJacobianPose2Trivial", testJacobianPose2Trivial),
  ]

  /// tests a simple identity Jacobian for Pose2
  func testJacobianPose2Identity() {
    let wT1 = Pose2(1, 0, 3.1415926 / 2.0), wT2 = Pose2(1, 0, 3.1415926 / 2.0)
    let pts: [Pose2] = [wT1, wT2]

    let f: @differentiable(_ pts: [Pose2]) -> Double = { (_ pts: [Pose2]) -> Double in
      let d = between(pts[0], pts[1])

      return d.rot.theta * d.rot.theta + d.t.x * d.t.x + d.t.y * d.t.y
    }

    let j = jacobian(of: f, at: pts)
    // print("J(f) = \(j[0].base as AnyObject)")
    for item in j[0] {
      XCTAssertEqual(item, Pose2.TangentVector.zero)
    }
  }

  func testJacobianPose2Trivial() {
    // Values taken from GTSAM `testPose2.cpp`
    let wT1 = Pose2(1, 2, .pi/2.0), wT2 = Pose2(-1, 4, .pi)
    let pts: [Pose2] = [wT1, wT2]

    let f: @differentiable(_ pts: [Pose2]) -> Pose2 = { (_ pts: [Pose2]) -> Pose2 in
      let d = between(pts[0], pts[1])

      return d
    }

    let j = jacobian(of: f, at: pts)

    let expected = Tensor<Double>([
      [0.0, -1.0, -2.0, 1.0, 0.0, 0.0],
      [1.0,  0.0, -2.0, 0.0, 1.0, 0.0],
      [0.0,  0.0, -1.0, 0.0, 0.0, 1.0]
    ])

    assertEqual(Tensor<Double>(matrixRows: j), expected, accuracy: 1e-10)
}

  /// tests the Jacobian of a 2D function
  func testJacobian2D() {
    let p1 = Point2(0, 1), p2 = Point2(0,0), p3 = Point2(0,0);
    let pts: [Point2] = [p1, p2, p3]

    // TODO(fan): Find a better way to do this
    // If we remove the type we will have:
    // a '@differentiable' function can only be formed from
    // a reference to a 'func' or a literal closure
    let f: @differentiable(_ pts: [Point2]) -> Point2 = { (_ pts: [Point2]) -> Point2 in
      let d = pts[1] - pts[0]

      return d
    }

    let j = jacobian(of: f, at: pts)
    
    // print("j(f) = \(j as AnyObject)")

    // print("Point2.TangentVector.basisVectors() = \(Point2.TangentVector.basisVectors() as AnyObject)")
    
    /* Example output:
      J(f) = [
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
    
    let expected: [Array<Point2>.TangentVector] = [
        [Point2.TangentVector(x: -1.0, y: 0.0), Point2.TangentVector(x: 1.0, y: 0.0), Point2.TangentVector(x: 0.0, y: 0.0)],
        [Point2.TangentVector(x: 0.0, y: -1.0), Point2.TangentVector(x: 0.0, y: 1.0), Point2.TangentVector(x: 0.0, y: 0.0)]
    ]
    /*
    print("J_f(p) = [")
    for c in j {
      print("[")
      for r in c {
        print(r.recursivelyAllKeyPaths(to:Double.self).map {r[keyPath: $0]})
        print(",")
      }
      print("]")
    }
    print("]")
    */
    XCTAssertEqual(expected, j)
  }
}
