import SwiftFusion
import TensorFlow
import XCTest

/// Asserts that `x` and `y` have the same shape and that their values have absolute difference
/// less than `accuracy`.
func assertEqual<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>,
  _ y: Tensor<T>,
  accuracy: T,
  message: String = "",
  file: StaticString = #file,
  line: UInt = #line
) {
  guard x.shape == y.shape else {
    XCTFail(
      "\(message)shape mismatch: \(x.shape) is not equal to \(y.shape)",
      file: file,
      line: line
    )
    return
  }
  XCTAssert(
    abs(x - y).max().scalarized() < accuracy,
    "\(message)value mismatch:\n\(x)\nis not equal to\n\(y)\nwith accuracy \(accuracy)",
    file: file,
    line: line
  )
}

/// TODO(document)
func assertFactor<F: TensorFactor>(
  _ factor: F,
  at value: Tensor<Double>,
  expectedError: Tensor<Double>,
  expectedJacobian: Tensor<Double>,
  accuracy: Double,
  file: StaticString = #file,
  line: UInt = #line
) {
  let actualError = factor.error(tensorValues: value)
  assertEqual(
    actualError,
    expectedError,
    accuracy: accuracy,
    message: "unexpected `error`: ",
    file: file,
    line: line
  )

  let (actualError2, actualPullback) = valueWithPullback(at: value, in: factor.error)
  assertEqual(
    actualError2,
    expectedError,
    accuracy: accuracy,
    message: "unexpected `error` from `errorWithPullback`: ",
    file: file,
    line: line
  )
  let expectedJacobianRowCount = expectedJacobian.shape[0]
  let actualJacobianRows = (0..<expectedJacobianRowCount).map { rowIndex -> Tensor<Double> in
    var basisVector = Tensor<Double>(zeros: [expectedJacobianRowCount])
    basisVector[rowIndex] = Tensor(1)
    return actualPullback(basisVector)
  }
  let actualJacobian = Tensor(stacking: actualJacobianRows)
  assertEqual(
    actualJacobian,
    expectedJacobian,
    accuracy: accuracy,
    message: "unexpected jacobian: ",
    file: file,
    line: line
  )
}
