import Foundation
import TensorFlow
import XCTest

import SwiftFusion

class JacobianTests: XCTestCase {
  /// tests the Jacobian of a 2D function
  func testJacobian2D() {
    let p1 = Vector2(0, 1), p2 = Vector2(0,0), p3 = Vector2(0,0);
    let pts: [Vector2] = [p1, p2, p3]

    // TODO(fan): Find a better way to do this
    // If we remove the type we will have:
    // a '@differentiable' function can only be formed from
    // a reference to a 'func' or a literal closure
    let f: @differentiable(_ pts: [Vector2]) -> Vector2 = { (_ pts: [Vector2]) -> Vector2 in
      let d = pts[1] - pts[0]

      return d
    }

    let j = jacobian(of: f, at: pts)
    
    // print("j(f) = \(j as AnyObject)")

    // print("Vector2.basisVectors() = \(Vector2.basisVectors() as AnyObject)")
    
    /* Example output:
      J(f) = [
      [ [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.0] ]
      [ [0.0, -1.0],
        [0.0, 1.0],
        [0.0, 0.0] ]
      ]
     So this is 2x3 but the data type is Vector2.
     In "normal" Jacobian notation, we should have a 2x6.
     [ [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
       [0.0, -1.0, 0.0, 1.0, 0.0, 0.0] ]
    */
    
    let expected: [Array<Vector2>.TangentVector] = [
        [Vector2(-1.0, 0.0), Vector2(1.0, 0.0), Vector2(0.0, 0.0)],
        [Vector2(0.0, -1.0), Vector2(0.0, 1.0), Vector2(0.0, 0.0)]
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
