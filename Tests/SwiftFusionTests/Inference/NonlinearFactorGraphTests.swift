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
    print("fg = \(fg)")
    print("fg_l = \(gfg)")
    print("bf1 = \(bf1)")
    print("bf1_l = \(bf1.linearize(val))")
    
    var vv = VectorValues()
    
    vv.insert(0, Tensor<Double>(shape:[3, 1], scalars: [1.0, 1.0, 0.0]))
    vv.insert(1, Tensor<Double>(shape:[3, 1], scalars: [1.0, 1.0, 3.14]))
    print("gfg(x) = \(gfg*vv)")
    print(bf1.linearize(val).jacobians[0].shape)
  }
}
