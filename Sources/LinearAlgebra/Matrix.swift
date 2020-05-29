/// Views `Columns` as the linear map obtained by horizontally stacking them in a matrix.
///
/// If there are `n` columns of type `V`, then `Matrix<Columns>` is the linear map from `V.Scalar^n`
/// to `V` obtained scaling each column by the corresponding element of `V.Scalar^n` and summing the
/// results.
///
/// For example, consider a matrix with 2 columns:
/// ```
/// let m = Matrix(
///   Vector(1, 0, 1), // first column
///   Vector(0, 1, 1)  // second column
/// )
/// ```
/// `m` is a linear map from `Vector2` to `Vector3`. `m(Vector2(1, 2))` multiplies the first column
/// by `1`, multiplies the second column by `2`, and sums the results, to get `Vector3(1, 2, 3)`.
public struct Matrix<Columns> {
  /// The columns.
  var columns: Columns

  /// Creates a matrix with `columns`.
  public init(_ columns: Columns) {
    self.columns = columns
  }
}

extension Matrix: LinearMap where Columns: CollectionOfVectors {
  /// The column.
  public typealias Column = Columns.Element

  /// The input of the linear map.
  public typealias Input = Vector<Columns.CollectionOfScalars>

  /// The output of the linear map.
  public typealias Output = Column

  /// Returns the result of the linear map applied to `x`.
  ///
  /// Precondition: `x` has the same number of scalars as there are columns in this matrix.
  public func apply(_ x: Input) -> Output {
    precondition(x.scalars.count == columns.count)

    // If there are no columns, then the output is zero.
    guard columns.count > 0 else { return Output.zero }

    // Initialize the result by applying the first column to the first input component.
    var inputIterator = x.scalars.makeIterator()
    var columnIterator = columns.makeIterator()
    var result = inputIterator.next()! * columnIterator.next()!

    // Accumulate the rest of the result by applying subsequent columns to subsequent input
    // components.
    while let column = columnIterator.next() {
      result += inputIterator.next()! * column
    }
    return result
  }

  /// Returns the matrix product of `self` and `other`.
  ///
  /// This applies `self` to each column of `other` and stacks the results horizontally into a
  /// result matrix.
  ///
  /// Precondition: The number of columns of `self` is equal to the number of rows of `other`.
  public func compose<
    OtherColumns: Collection,
    ResultColumns: FixedSizeArray
  >(
    _ other: Matrix<OtherColumns>
  ) -> Matrix<ResultColumns>
  where
    Matrix<OtherColumns>.Column == Input,
    ResultColumns.Element == Output
  {
    let resultColumns = ResultColumns(other.columns.lazy.map { self.apply($0) })
    return Matrix<ResultColumns>(resultColumns)
  }
  
  /// Returns the matrix-vector product of `matrix` with `vector`.
  public static func * (_ matrix: Self, _ vector: Input) -> Output {
    return matrix.apply(vector)
  }
  
  /// Returns the matrix product of `matrix1` with `matrix2`.
  public static func * <
    OtherColumns: Collection,
    ResultColumns: FixedSizeArray
  >(_ matrix1: Self, _ matrix2: Matrix<OtherColumns>) -> Matrix<ResultColumns>
  where
    Matrix<OtherColumns>.Column == Input,
    ResultColumns.Element == Output
  {
    return matrix1.compose(matrix2)
  }
}

extension Matrix: Equatable where Columns: Equatable {}

/// A collection of vectors.
///
/// This protocol allows us to determine the input type of a matrix. For example, the input type
/// of `Matrix<Array3<Vector2>>` is `Vector3`.
public protocol CollectionOfVectors: Collection where Element: VectorProtocol {
  /// The collection of scalars.
  associatedtype CollectionOfScalars: Equatable & FixedSizeArray
  where CollectionOfScalars.Element == Element.Scalar
}

// MARK: - "Generated Code"

extension Array1: CollectionOfVectors where Element: VectorProtocol {
  public typealias CollectionOfScalars = Array1<Element.Scalar>
}
extension Array2: CollectionOfVectors where Element: VectorProtocol {
  public typealias CollectionOfScalars = Array2<Element.Scalar>
}
extension Array3: CollectionOfVectors where Element: VectorProtocol {
  public typealias CollectionOfScalars = Array3<Element.Scalar>
}

extension Matrix {
  /// Creates a matrix with 1 column.
  public init<Scalars>(_ column0: Vector<Scalars>) where Columns == Array1<Vector<Scalars>> {
    self.columns = Array1(column0)
  }
  
  /// Creates a matrix with 2 columns.
  public init<Scalars>(
    _ column0: Vector<Scalars>,
    _ column1: Vector<Scalars>
  ) where Columns == Array2<Vector<Scalars>> {
    self.columns = Array2(column0, column1)
  }
  
  /// Creates a matrix with 3 columns.
  public init<Scalars>(
    _ column0: Vector<Scalars>,
    _ column1: Vector<Scalars>,
    _ column2: Vector<Scalars>
  ) where Columns == Array3<Vector<Scalars>> {
    self.columns = Array3(column0, column1, column2)
  }
}
