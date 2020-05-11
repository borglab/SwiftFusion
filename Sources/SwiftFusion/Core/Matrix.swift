import TensorFlow

/// A dynamically sized matrix.
public struct Matrix: Equatable {
  /// The scalars in the matrix, in row major order.
  public fileprivate(set) var scalars: [Double]

  /// The number of rows and columns in the matrix.
  public fileprivate(set) var rowCount, columnCount: Int
}

/// Initializers.
extension Matrix {
  /// Creates a matrix with `scalars` in row major order with shape `rowCount` x `columnCount`.
  public init(_ scalars: [Double], rowCount: Int, columnCount: Int) {
    precondition(scalars.count == rowCount * columnCount)
    self.scalars = scalars
    self.rowCount = rowCount
    self.columnCount = columnCount
  }

  /// Creates a matrix stacking `rows`.
  public init(stacking rows: [Vector]) {
    guard rows.count > 0 else {
      self.init([], rowCount: 0, columnCount: 0)
      return
    }
    self.init([], rowCount: 0, columnCount: rows[0].dimension)
    for row in rows {
      self.append(row: row)
    }
  }

  /// Creates an identity matrix with the given `dimension`.
  public init(eye dimension: Int) {
    self.init([], rowCount: 0, columnCount: dimension)
    self.reserveCapacity(dimension * dimension)
    for i in 0..<dimension {
      var scalars = Array(repeating: Double(0), count: dimension)
      scalars[i] = 1
      self.append(row: Vector(scalars))
    }
  }
}

/// Subscripts.
extension Matrix {
  // Returns the `(row, column)` entry of the matrix.
  public subscript(row: Int, column: Int) -> Double {
    return scalars[row * columnCount + column]
  }
}

/// Elementwise operations.
extension Matrix {
  public static func *= (_ lhs: inout Matrix, _ rhs: Double) {
    for index in lhs.scalars.indices {
      lhs.scalars[index] *= rhs
    }
  }

  public static func * (_ lhs: Double, _ rhs: Matrix) -> Matrix {
    var result = rhs
    result *= lhs
    return result
  }

  public static func * (_ lhs: Matrix, _ rhs: Double) -> Matrix {
    var result = lhs
    result *= rhs
    return result
  }
}

/// Mutations.
extension Matrix {
  /// Appends `row` at the bottom of the matrix.
  ///
  /// Precondition `row.dimension == self.columnCount`.
  public mutating func append(row: Vector) {
    precondition(row.dimension == self.columnCount)
    scalars.append(contentsOf: row.scalars)
    rowCount += 1
  }

  /// Allocates space for `n` scalars in the scalar storage, so that no reallocations are
  /// necessary while adding up to that many elements.
  public mutating func reserveCapacity(_ n: Int) {
    scalars.reserveCapacity(n)
  }
}

/// Returns the matrix-vector product of `lhs` and `rhs`.
public func matvec(_ lhs: Matrix, transposed: Bool = false, _ rhs: Vector) -> Vector {
  precondition(rhs.dimension == (transposed ? lhs.rowCount : lhs.columnCount))
  var result = Vector(zeros: transposed ? lhs.columnCount : lhs.rowCount)
  for i in 0..<lhs.rowCount {
    for j in 0..<lhs.columnCount {
      result.scalars[transposed ? j : i] += lhs[i, j] * rhs.scalars[transposed ? i : j]
    }
  }
  return result
}
