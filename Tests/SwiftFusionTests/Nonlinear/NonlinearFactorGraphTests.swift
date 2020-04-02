import Foundation
import TensorFlow
import XCTest

import SwiftFusion

class NonlinearFactorGraphTests: XCTestCase {
  static var allTests = [
    ("test1", test1),
  ]

  func test1() {
    func priorFactor(_ x: Pose2) -> @differentiable (Pose2) -> Pose2 {
      { between(x, $0) }
    }

    func betweenFactor(_ delta: Pose2) -> @differentiable (Pose2, Pose2) -> Pose2 {
      { between($1, delta * $0) }
    }

    let priorNoise = NoiseModel(1, 1, 1)

    var graph = NonlinearFactorGraph()
    graph.add(0, priorFactor(Pose2(0, 0, 0)), priorNoise)
    graph.add(0, 1, betweenFactor(Pose2(1, 0, 0)), priorNoise)

    var values = [Pose2(0, 0, 1), Pose2(1, 1, 0)]

    for step in 0..<2 {
      print("STEP \(step)")
      print(values)
      print(graph.loss(at: values))
      print(graph.errors(at: values))
      print(graph.linearize(at: values))

      let m = graph.linearize(at: values)
      let errors = graph.errors(at: values)

      // Terribly inefficient computation of OLS solution.
      // Doesn't use the noise models yet.
      let pseudoinverse = matmul(_Raw.matrixInverse(matmul(m.transposed(), m)), m.transposed())
      let step = matmul(pseudoinverse, -errors)

      var typedStep = Array<Pose2.TangentVector>.TangentVector([])
      for index in values.indices {
        typedStep.base.append(Pose2.TangentVector(
          step[3 * index, 0].scalarized(),
          step[3 * index + 1, 0].scalarized(),
          step[3 * index + 2, 0].scalarized()
        ))
      }

      values.move(along: typedStep)
      print(values)
    }
  }
}
