// This file tests for the CGLS optimizer

import SwiftFusion
import PenguinStructures
import TensorFlow
import XCTest

final class CGLSTests: XCTestCase {  
  /// test convergence for a simple gaussian factor graph
  func testCGLSSolver() {
    let gfg = SimpleGaussianFactorGraph.create()
    
    var optimizer = GenericCGLS(precision: 1e-7, max_iteration: 10)
    var x: VariableAssignments = SimpleGaussianFactorGraph.zeroDelta()
    optimizer.optimize(gfg: gfg, initial: &x)
    
    let expected = SimpleGaussianFactorGraph.correctDelta()
    
    for k in [SimpleGaussianFactorGraph.x1ID, SimpleGaussianFactorGraph.x2ID] {
      assertEqual(x[k].flatTensor, expected[k].flatTensor, accuracy: 1e-6)
    }
  }
}
