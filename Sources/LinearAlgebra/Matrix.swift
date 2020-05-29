/// Views `Columns` as the linear map obtained by horizontally stacking them in a matrix.
///
/// If there are `n` columns, and if each column is a linear map from `A` to `B`, then
/// `Matrix<Columns>` is the linear map from `A^n` to `B` obtained by applying each column to its
/// corresponding element of `A^n` and summing the results.
///
/// For example, consider a matrix with 2 columns:
/// ```
/// let m = Matrix3x2(
///   Vector3d(1, 0, 1), // first column
///   Vector3d(0, 1, 1)  // second column
/// )
/// ```
/// We view each `Vector3d` as a linear map from `ScalarVector<Double>` to `Vector3d` via scalar
/// multiplication. So `m` is a linear map from `ScalarVector<Double>^2` (aka `Vector2d`) to
/// `Vector3d`.
/// `m(Vector2d(1, 2))` multiplies the first column by `1`, multiplies the second column by `2`,
/// and sums the results, to get `Vector3d(1, 2, 3)`.
public struct Matrix<Columns> {
  /// The columns.
  var columns: Columns

  /// Creates a matrix with `columns`.
  public init(_ columns: Columns) {
    self.columns = columns
  }
}

extension Matrix: LinearMap where Columns: CollectionOfLinearMaps {
  /// The column.
  public typealias Column = Columns.Element

  /// The input of the linear map.
  ///
  /// This is a collection of column inputs because the matrix linear map works by applying each
  /// column to the corresponding element of the input.
  public typealias Input = Columns.CollectionOfElementInputs

  /// The output of the linear map.
  ///
  /// This is the same as the column output because the matrix linear map works by summing outputs
  /// from each column.
  public typealias Output = Column.Output

  /// Returns the result of the linear map applied to `x`.
  ///
  /// Precondition: `x` has the same number of elements as there are columns in this matrix.
  public func callAsFunction(_ x: Input) -> Output {
    precondition(x.count == columns.count)

    // If there are no columns, then the output is zero.
    guard columns.count > 0 else { return Output.zero }

    // Initialize the result by applying the first column to the first input component.
    var inputIterator = x.makeIterator()
    var columnIterator = columns.makeIterator()
    var result = columnIterator.next()!(inputIterator.next()!)

    // Accumulate the rest of the result by applying subsequent columns to subsequent input
    // components.
    while let column = columnIterator.next() {
      result += column(inputIterator.next()!)
    }
    return result
  }

  /// Returns the matrix product of `self` and `other`.
  ///
  /// This applies `self` to each column of `other` and stacks the results horizontally into a
  /// result matrix.
  ///
  /// Precondition: The number of columns of `self` is equal to the number of rows of `other`.
  public func matmul<
    OtherColumns: Collection,
    ResultColumns: FixedSizeArray
  >(
    _ other: Matrix<OtherColumns>
  ) -> Matrix<ResultColumns>
  where
    Matrix<OtherColumns>.Column == Input,
    ResultColumns.Element == Output
  {
    let resultColumns = ResultColumns(other.columns.lazy.map { self($0) })
    return Matrix<ResultColumns>(resultColumns)
  }
}

extension Matrix: Equatable where Columns: Equatable {}

/// A collection of linear maps.
///
/// This protocol allows us to determine the input type of a matrix. For example, the input type
/// of `Matrix<Array3<Vector2f>>` is `Vector3f`.
public protocol CollectionOfLinearMaps: Collection where Element: LinearMap {
  /// The collection of element inputs.
  associatedtype CollectionOfElementInputs: Collection & Vector
  where CollectionOfElementInputs.Element == Element.Input
}

// MARK: - "Generated Code"

extension Array1: CollectionOfLinearMaps where Element: LinearMap {
  public typealias CollectionOfElementInputs = Array1<Element.Input>
}
extension Array2: CollectionOfLinearMaps where Element: LinearMap {
  public typealias CollectionOfElementInputs = Array2<Element.Input>
}
extension Array3: CollectionOfLinearMaps where Element: LinearMap {
  public typealias CollectionOfElementInputs = Array3<Element.Input>
}
