// This file tests for the CGLS optimizer

import SwiftFusion
import TensorFlow
import XCTest

final class CGLSTests: XCTestCase {
  func Vector2_t(_ x: Double, _ y: Double) -> Tensor<Double> {
    let t = Tensor<Double>(shape: [2, 1], scalars: [x, y])
    return t
  }
  
  /// test convergence for a simple gaussian factor graph
  func testCGLSSolver() {
    let gfg = createSimpleGaussianFactorGraph()
    
    let optimizer = CGLS(precision: 0.01, max_iteration: 10)
    var x: VectorValues = createZeroDelta()
    optimizer.optimize(gfg: gfg, initial: &x)
    
    let expected = createCorrectDelta()
    // Test condition: P_5 should be identical to P_1 (close loop)
    // assertEqual(x[0], createCorrectDelta()[0], accuracy: 0.1)
    print("final x = \(x), correct = \(expected)")
    for (k, _) in x.indices {
      assertEqual(x[k], expected[k], accuracy: 1e-6)
    }
  }

  static var allTests = [
    ("testCGLSSolver", testCGLSSolver),
  ]
}
