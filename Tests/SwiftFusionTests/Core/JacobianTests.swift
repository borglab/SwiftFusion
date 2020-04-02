import Foundation
import TensorFlow
import XCTest

import SwiftFusion

class JacobianTests: XCTestCase {
  static var allTests = [
    ("testJacobian2D", testJacobian2D)
  ]

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
