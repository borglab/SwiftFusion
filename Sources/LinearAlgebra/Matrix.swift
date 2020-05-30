/// Views `Columns` as the linear map obtained by horizontally stacking them in a matrix.
///
/// If there are `n` columns of type `Output`, then the linear map acts on `Input` by scaling each
/// column by the corresponding element of `Input` and summing the results.
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
public struct Matrix<Columns: Collection, Input: VectorProtocol & Collection>
where Columns.Element: VectorProtocol, Columns.Element.Scalar == Input.Scalar,
      Input.Index == Columns.Index, Input.Element == Input.Scalar
{
  /// The columns.
  var columns: Columns

  /// Creates a matrix with `columns`.
  public init(_ columns: Columns) {
    self.columns = columns
  }
}

extension Matrix: LinearMap {
  /// The output of the linear map.
  public typealias Output = Columns.Element
  
  /// Returns the result of the linear map applied to `x`.
  ///
  /// Precondition: `x` has the same number of scalars as there are columns in `self`.
  public func forward(_ x: Input) -> Output {
    precondition(x.count == columns.count)

    // If there are no columns, then the output is zero.
    guard columns.count > 0 else { return Output.zero }

    // Initialize the result by applying the first column to the first input component.
    var inputIterator = x.makeIterator()
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
  /// Precondition: Each column in `other` has the same number of scalars as there are columns in
  /// `self`.
  public func compose<OtherColumns, OtherInput, ResultColumns: InitializableFromCollection>(
    _ other: Matrix<OtherColumns, OtherInput>
  ) -> Matrix<ResultColumns, OtherInput>
  where OtherColumns.Element == Input, ResultColumns.Element == Output {
    let resultColumns = ResultColumns(other.columns.lazy.map { self.forward($0) })
    return Matrix<ResultColumns, OtherInput>(resultColumns)
  }
  
  /// Returns the matrix-vector product of `matrix` with `vector`.
  public static func * (_ matrix: Self, _ vector: Input) -> Output {
    return matrix.forward(vector)
  }
  
  /// Returns the matrix product of `matrix1` with `matrix2`.
  public static func * <OtherColumns, OtherInput, ResultColumns: InitializableFromCollection>(
    _ matrix1: Self,
    _ matrix2: Matrix<OtherColumns, OtherInput>
  ) -> Matrix<ResultColumns, OtherInput>
  where OtherColumns.Element == Input, ResultColumns.Element == Output {
    return matrix1.compose(matrix2)
  }
}

extension Matrix: Equatable where Columns: Equatable {}

// MARK: - "Generated Code"

extension Matrix {
  /// Creates a matrix with 1 column.
  public init<Scalar, Column>(
    _ column0: Column
  ) where Columns == Array1<Column>,
          Input == Vector<Array1<Scalar>>
  {
    self.columns = Array1(column0)
  }
  
  /// Creates a matrix with 2 columns.
  public init<Scalar, Column>(
    _ column0: Column,
    _ column1: Column
  ) where Columns == Array2<Column>,
          Input == Vector<Array2<Scalar>>
  {
    self.columns = Array2(column0, column1)
  }
  
  /// Creates a matrix with 3 columns.
  public init<Scalar, Column>(
    _ column0: Column,
    _ column1: Column,
    _ column2: Column
  ) where Columns == Array3<Column>,
          Input == Vector<Array3<Scalar>>
  {
    self.columns = Array3(column0, column1, column2)
  }
}
