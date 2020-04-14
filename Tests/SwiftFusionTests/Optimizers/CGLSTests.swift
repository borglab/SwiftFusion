// This file tests for the CGLS optimizer

import SwiftFusion
import TensorFlow
import XCTest

final class CGLSTests: XCTestCase {  
  /// test convergence for a simple gaussian factor graph
  func testCGLSSolver() {
    let gfg = createSimpleGaussianFactorGraph()
    
    let optimizer = CGLS(precision: 0.01, max_iteration: 10)
    var x: VectorValues = createZeroDelta()
    optimizer.optimize(gfg: gfg, initial: &x)
    
    let expected = createCorrectDelta()
    
    for (k, _) in x.indices {
      assertEqual(x[k], expected[k], accuracy: 1e-6)
    }
  }

  static var allTests = [
    ("testCGLSSolver", testCGLSSolver),
  ]
}
