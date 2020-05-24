public struct Matrix3: Differentiable {
  var s00, s01, s02: Double
  var s10, s11, s12: Double
  var s20, s21, s22: Double

  public var columnCount: Int {
    3
  }
  
  public var rowCount: Int {
    3
  }
  
  public
  init(
    _ s00: Double, _ s01: Double, _ s02: Double,
    _ s10: Double, _ s11: Double, _ s12: Double,
    _ s20: Double, _ s21: Double, _ s22: Double
  ) {
    self.s00 = s00
    self.s01 = s01
    self.s02 = s02
    self.s10 = s10
    self.s11 = s11
    self.s12 = s12
    self.s20 = s20
    self.s21 = s21
    self.s22 = s22
  }

  public
  init(stacking row0: Vector3, _ row1: Vector3, _ row2: Vector3) {
    self.init(
      row0.x, row0.y, row0.z,
      row1.x, row1.y, row1.z,
      row2.x, row2.y, row2.z
    )
  }

  public static func matvec(_ lhs: Matrix3, _ rhs: Vector3) -> Vector3 {
    return Vector3(
      lhs.s00 * rhs.x + lhs.s01 * rhs.y + lhs.s02 * rhs.z,
      lhs.s10 * rhs.x + lhs.s11 * rhs.y + lhs.s12 * rhs.z,
      lhs.s20 * rhs.x + lhs.s21 * rhs.y + lhs.s22 * rhs.z
    )
  }

  public static func matvec(transposed lhs: Matrix3, _ rhs: Vector3) -> Vector3 {
    return Vector3(
      lhs.s00 * rhs.x + lhs.s10 * rhs.y + lhs.s20 * rhs.z,
      lhs.s01 * rhs.x + lhs.s11 * rhs.y + lhs.s21 * rhs.z,
      lhs.s02 * rhs.x + lhs.s12 * rhs.y + lhs.s22 * rhs.z
    )
  }
  
  /// Returns the matrix-vector product of `lhs` and `rhs`.
  public func matmul(_ lhs: Matrix3, _ rhs: Matrix3) -> Matrix3 {
    precondition(rhs.rowCount == lhs.columnCount)
    var result = Matrix3(repeating: 0, rowCount: lhs.rowCount, columnCount: rhs.columnCount)
    for line in 0..<rhs.columnCount {
      for i in 0..<lhs.rowCount {
        for j in 0..<lhs.columnCount {
          result.scalars[i * result.columnCount + line] += lhs[i, j] * rhs.scalars[j * result.columnCount + line]
        }
      }
    }
    return result
  }
}
