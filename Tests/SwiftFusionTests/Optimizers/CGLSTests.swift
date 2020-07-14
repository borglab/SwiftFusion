// This file tests for the CGLS optimizer

import SwiftFusion
import PenguinStructures
import TensorFlow
import XCTest

final class CGLSTests: XCTestCase {  
  /// test convergence for a simple gaussian factor graph
  func testCGLSSolver() {
    let gfg = SimpleOldGaussianFactorGraph.create()
    
    let optimizer = CGLS(precision: 1e-7, max_iteration: 10)
    var x: VectorValues = SimpleOldGaussianFactorGraph.zeroDelta()
    optimizer.optimize(gfg: gfg, initial: &x)
    
    let expected = SimpleOldGaussianFactorGraph.correctDelta()
    
    for k in x.keys {
      assertEqual(x[k].tensor, expected[k].tensor, accuracy: 1e-6)
    }
  }
}
