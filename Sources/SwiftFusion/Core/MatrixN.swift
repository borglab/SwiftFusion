public struct Matrix3: Differentiable & Equatable & KeyPathIterable {
  public var s00, s01, s02: Double
  public var s10, s11, s12: Double
  public var s20, s21, s22: Double

  public var columnCount: Int {
    3
  }
  
  public var rowCount: Int {
    3
  }
  
  public static var Identity: Matrix3 {
    return Matrix3(1, 0, 0, 0, 1, 0, 0, 0, 1)
  }
  
  @differentiable
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

  @differentiable
  public
  init(stacking row0: Vector3, _ row1: Vector3, _ row2: Vector3) {
    self.init(
      row0.x, row0.y, row0.z,
      row1.x, row1.y, row1.z,
      row2.x, row2.y, row2.z
    )
  }

  @differentiable
  public
  init(columns col0: Vector3, _ col1: Vector3, _ col2: Vector3) {
    self.init(
      col0.x, col1.x, col2.x,
      col0.y, col1.y, col2.y,
      col0.z, col1.z, col2.z
    )
  }
  
  @differentiable
  public static func + (_ lhs: Matrix3, _ rhs: Matrix3) -> Matrix3 {
    Matrix3(lhs.s00 + rhs.s00, lhs.s01 + rhs.s01, lhs.s02 + rhs.s02,
            lhs.s10 + rhs.s10, lhs.s11 + rhs.s11, lhs.s12 + rhs.s12,
            lhs.s20 + rhs.s20, lhs.s21 + rhs.s21, lhs.s22 + rhs.s22)
  }
  
  @differentiable
  public static func / (_ lhs: Matrix3, _ rhs: Double) -> Matrix3 {
    Matrix3(lhs.s00 / rhs, lhs.s01 / rhs, lhs.s02 / rhs,
            lhs.s10 / rhs, lhs.s11 / rhs, lhs.s12 / rhs,
            lhs.s20 / rhs, lhs.s21 / rhs, lhs.s22 / rhs)
  }
  
  @differentiable
  public static func * (_ scalar: Double, _ mat: Matrix3) -> Matrix3 {
    Matrix3(scalar * mat.s00, scalar * mat.s01, scalar * mat.s02,
            scalar * mat.s10, scalar * mat.s11, scalar * mat.s12,
            scalar * mat.s20, scalar * mat.s21, scalar * mat.s22)
  }
  
  @differentiable
  public func transposed() -> Matrix3 {
    Matrix3(s00, s10, s20,
            s01, s11, s21,
            s02, s12, s22)
  }
}


/// M * v where v is 3x1
@differentiable
public func matvec(_ lhs: Matrix3, _ rhs: Vector3) -> Vector3 {
  return Vector3(
    lhs.s00 * rhs.x + lhs.s01 * rhs.y + lhs.s02 * rhs.z,
    lhs.s10 * rhs.x + lhs.s11 * rhs.y + lhs.s12 * rhs.z,
    lhs.s20 * rhs.x + lhs.s21 * rhs.y + lhs.s22 * rhs.z
  )
}

/// M * v where v is 1x3
@differentiable
public func matvec(transposed lhs: Matrix3, _ rhs: Vector3) -> Vector3 {
  return Vector3(
    lhs.s00 * rhs.x + lhs.s10 * rhs.y + lhs.s20 * rhs.z,
    lhs.s01 * rhs.x + lhs.s11 * rhs.y + lhs.s21 * rhs.z,
    lhs.s02 * rhs.x + lhs.s12 * rhs.y + lhs.s22 * rhs.z
  )
}

/// Returns the matrix-vector product of `lhs` and `rhs`.
@differentiable
public func matmul(_ lhs: Matrix3, _ rhs: Matrix3) -> Matrix3 {
  precondition(rhs.rowCount == lhs.columnCount)
  let v1 = matvec(lhs, Vector3(rhs.s00, rhs.s10, rhs.s20))
  let v2 = matvec(lhs, Vector3(rhs.s01, rhs.s11, rhs.s21))
  let v3 = matvec(lhs, Vector3(rhs.s02, rhs.s12, rhs.s22))
  
  return Matrix3(columns: v1, v2, v3)
}
