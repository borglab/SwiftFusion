// This file tests for the CGLS optimizer

import SwiftFusion
import TensorFlow
import XCTest

final class CGLSTests: XCTestCase {  
  /// test convergence for a simple gaussian factor graph
  func testCGLSSolver() {
    let gfg = SimpleGaussianFactorGraph.create()
    
    let optimizer = CGLS(precision: 1e-7, max_iteration: 10)
    var x: VectorValues = SimpleGaussianFactorGraph.zeroDelta()
    optimizer.optimize(gfg: gfg, initial: &x)
    
    let expected = SimpleGaussianFactorGraph.correctDelta()
    
    for k in x.keys {
      assertEqual(x[k].tensor, expected[k].tensor, accuracy: 1e-6)
    }
  }
}
