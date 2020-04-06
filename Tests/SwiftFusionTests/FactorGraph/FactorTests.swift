import TensorFlow
import XCTest

import SwiftFusion

final class FactorTests: XCTestCase {
  /// Test that `Factor` computes the correct error and jacobian for a simple "identity" factor.
  func testFactorIdentity() {
    struct IdentityFactor: Factor, TensorFactor {
      @differentiable func error(_ input: Vector2) -> Vector2 {
        input
      }
    }
    for _ in 0..<10 {
      let value = Vector2(Tensor<Double>(randomNormal: [2]))
      assertFactor(
        IdentityFactor(),
        at: value.tensor,
        expectedError: value.tensor,
        expectedJacobian: eye(rowCount: 2),
        accuracy: 1e-10
      )
    }
  }

  /// Test that `Factor` computes the correct error and jacobian for a simple "difference" factor.
  func testFactorDifference() {
    struct DifferenceFactor: Factor, TensorFactor {
      typealias Input = TensorConvertiblePair<Vector2, Vector2>
      typealias Output = Vector2
      @differentiable func error(_ input: TensorConvertiblePair<Vector2, Vector2>) -> Vector2 {
        input.a - input.b
      }
    }
    for _ in 0..<10 {
      let value = DifferenceFactor.Input(
        Vector2(Tensor<Double>(randomNormal: [2])),
        Vector2(Tensor<Double>(randomNormal: [2]))
      )
      assertFactor(
        DifferenceFactor(),
        at: value.tensor,
        expectedError: (value.a - value.b).tensor,
        expectedJacobian: Tensor(concatenating: [eye(rowCount: 2), -eye(rowCount: 2)], alongAxis: 1),
        accuracy: 1e-10
      )
    }
  }


  static var allTests = [
    ("testFactorIdentity", testFactorIdentity),
    ("testFactorDifference", testFactorDifference)
  ]
}
