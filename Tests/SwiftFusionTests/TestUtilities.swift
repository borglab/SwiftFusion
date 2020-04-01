import TensorFlow
import XCTest

extension Tensor {
  init<T: KeyPathIterable>(matrixRows: [T]) {
    self.init(matrixRows.map { row in
      Tensor(row.recursivelyAllKeyPaths(to: Scalar.self).map { row[keyPath: $0] })
    })
  }
}

func assertEqual<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>, _ y: Tensor<T>, accuracy: T, file: StaticString = #file, line: UInt = #line
) {
  guard x.shape == y.shape else {
    XCTFail(
      "shape mismatch: \(x.shape) is not equal to \(y.shape)",
      file: file,
      line: line
    )
    return
  }
  XCTAssert(
    abs(x - y).max().scalarized() < accuracy,
    "value mismatch:\n\(x)\nis not equal to\n\(y)\nwith accuracy \(accuracy)",
    file: file,
    line: line
  )
}
