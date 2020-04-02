import TensorFlow

extension Tensor {
  /// Creates a matrix whose rows are the fields of type `Scalar` within `matrixRows`.
  ///
  /// For example, `Tensor<Double>(matrixRows: [Point2(1, 2), Point2(3, 4)])` creates the matrix:
  ///   1 2
  ///   3 4
  public init<T: KeyPathIterable>(matrixRows: [T]) {
    self.init(matrixRows.map { row in
      Tensor(row.recursivelyAllKeyPaths(to: Scalar.self).map { row[keyPath: $0] })
    })
  }
}

