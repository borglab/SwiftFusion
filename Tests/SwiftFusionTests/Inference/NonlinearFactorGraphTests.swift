import SwiftFusion
import TensorFlow
import XCTest

final class NonlinearFactorGraphTests: XCTestCase {
  /// test ATr
  func testBasicOps() {
    var fg = NonlinearFactorGraph()
    
    let bf1 = BetweenFactor(0, 1, Pose2(0.0,0.0, 0.0))
    
    fg += AnyNonlinearFactor(bf1)
    
    var val = Values()
    val.insert(0, AnyDifferentiable(Pose2(1.0, 1.0, 0.0)))
    val.insert(1, AnyDifferentiable(Pose2(1.0, 1.0, .pi)))
    
    let gfg = fg.linearize(val)
    
    var vv = VectorValues()
    
    vv.insert(0, Tensor<Double>(shape:[3, 1], scalars: [1.0, 1.0, 0.0]))
    vv.insert(1, Tensor<Double>(shape:[3, 1], scalars: [1.0, 1.0, 3.14]))
    
    let expected = Tensor<Double>(shape:[3, 1], scalars: [0.0, 0.0, -3.14])
    
    assertEqual((gfg * vv)[0], expected, accuracy: 1e-9)
  }
}
