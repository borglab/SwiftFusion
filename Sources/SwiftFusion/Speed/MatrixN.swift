protocol LinearFunction {
  associatedtype Input: EuclideanVectorSpace
  associatedtype Output: EuclideanVectorSpace
  func forward(_ x: Input) -> Output
  func transpose(_ y: Output) -> Input
}

struct Matrix3x3 {
  var s00, s01, s02: Double
  var s10, s11, s12: Double
  var s20, s21, s22: Double

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

  init(stacking row0: Vector3, _ row1: Vector3, _ row2: Vector3) {
    self.init(
      row0.x, row0.y, row0.z,
      row1.x, row1.y, row1.z,
      row2.x, row2.y, row2.z
    )
  }

  static func matvec(_ lhs: Matrix3x3, _ rhs: Vector3) -> Vector3 {
    return Vector3(
      lhs.s00 * rhs.x + lhs.s01 * rhs.y + lhs.s02 * rhs.z,
      lhs.s10 * rhs.x + lhs.s11 * rhs.y + lhs.s12 * rhs.z,
      lhs.s20 * rhs.x + lhs.s21 * rhs.y + lhs.s22 * rhs.z
    )
  }

  static func matvec(transposed lhs: Matrix3x3, _ rhs: Vector3) -> Vector3 {
    return Vector3(
      lhs.s00 * rhs.x + lhs.s10 * rhs.y + lhs.s20 * rhs.z,
      lhs.s01 * rhs.x + lhs.s11 * rhs.y + lhs.s21 * rhs.z,
      lhs.s02 * rhs.x + lhs.s12 * rhs.y + lhs.s22 * rhs.z
    )
  }
}

extension Matrix3x3: LinearFunction {
  func forward(_ x: Vector3) -> Vector3 {
    return Matrix3x3.matvec(self, x)
  }
  func transpose(_ y: Vector3) -> Vector3 {
    return Matrix3x3.matvec(transposed: self, y)
  }
}

func makeJacobian(transposing linear: (Vector3) -> Vector3) -> Matrix3x3 {
  let row0 = linear(Vector3(1, 0, 0))
  let row1 = linear(Vector3(0, 1, 0))
  let row2 = linear(Vector3(0, 0, 1))
  return Matrix3x3(stacking: row0, row1, row2)
}

func makeJacobian(transposing linear: (Vector3) -> (Vector3, Vector3)) -> (Matrix3x3, Matrix3x3) {
  let (row0a, row0b) = linear(Vector3(1, 0, 0))
  let (row1a, row1b) = linear(Vector3(0, 1, 0))
  let (row2a, row2b) = linear(Vector3(0, 0, 1))
  return (Matrix3x3(stacking: row0a, row1a, row2a), Matrix3x3(stacking: row0b, row1b, row2b))
}


// struct Matrix3x6 {
//   var a, b: Matrix3x3
// }
// 
// extension Matrix3x6: LinearFunction {
//   init(transposing linear: (Vector3) -> (Vector3, Vector3)) {
//     let row0 = linear(Vector3(1, 0, 0))
//     let row1 = linear(Vector3(0, 1, 0))
//     let row2 = linear(Vector3(0, 0, 1))
//     a = Matrix3x3(stacking: row0.value1, row1.value1, row2.value1)
//     b = Matrix3x3(stacking: row0.value2, row1.value2, row2.value2)
//   }
//   func forward(_ x: (Vector3, Vector3)) -> Vector3 {
//     return Matrix3x3.matvec(a, x.value1) + Matrix3x3.matvec(b, x.value2)
//   }
//   func transpose(_ y: Vector3) -> (Vector3, Vector3) {
//     return FactorVectorInput2(
//       value1: Matrix3x3.matvec(transposed: a, y),
//       value2: Matrix3x3.matvec(transposed: b, y)
//     )
//   }
// }
