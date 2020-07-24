// Copyright 2020 The SwiftFusion Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import PenguinStructures

public typealias Matrix2 = FixedSizeMatrix<Array2<Vector2>>
public typealias Matrix3 = FixedSizeMatrix<Array3<Vector3>>

/// A matrix whose dimensions are known at compile time.
///
/// Stored as a fixed-size array of rows, where the rows are `EuclideanVectorN`. For example,
/// - `FixedSizeMatrix<Array3<Vector2>>`: A 3x2 matrix, where each row is a `Vector2`.
/// - `FixedSizeMatrix<Array3<Tuple2<Vector2, Vector3>>`: A 3x5 matrix, where each row is a
///   `Tuple2<Vector2, Vector3>`.
public struct FixedSizeMatrix<Rows: Equatable & FixedSizeArray>
  where Rows.Element: EuclideanVectorN
{

  /// The elements of the matrix, stored as a fixed-size array of row vectors.
  public var rows: Rows

  /// The number of elements of each dimension of the matrix.
  ///
  /// `shape[0]` is the number of rows.
  /// `shape[1]` is the number of columns.
  public static var shape: Array2<Int> { return Array2(Rows.count, Rows.Element.dimension) }

  /// Accesses the element at `row`, `column`.
  public subscript(row: Int, column: Int) -> Double {
    _read {
      yield rows[row][column]
    }
    _modify {
      yield &rows[row][column]
    }
  }
}

// MARK: - Convenience initializers and static instances.

extension FixedSizeMatrix {
  /// Creates a matrix with `elements`, in row-major order.
  ///
  /// Requires: `elements.count == Self.shape[0] * Self.shape[1]`.
  public init<C: Collection>(_ elements: C) where C.Element == Double {
    precondition(elements.count == Self.shape[0] * Self.shape[1])
    self.rows = Rows((0..<Rows.count).lazy.map { i in
      Rows.Element(elements.dropFirst(i * Self.shape[1]))
    })
  }

  /// Creates a matrix with rows `r0`, `r1`, and `r2`.
  @differentiable
  public init(rows r0: Rows.Element, _ r1: Rows.Element, _ r2: Rows.Element) {
    // TODO: We can statically constrain `Rows` after TF-1292 is fixed.
    self.rows = Array3(r0, r1, r2) as! Rows
  }

  @derivative(of: init(rows:_:_:))
  @usableFromInline
  static func vjpInit(rows r0: Rows.Element, _ r1: Rows.Element, _ r2: Rows.Element) -> (
    value: Self,
    pullback: (Self) -> (Rows.Element, Rows.Element, Rows.Element)
  ) {
    return (Self(rows: r0, r1, r2), { v in (v.rows[0], v.rows[1], v.rows[2]) })
  }

  /// The identity matrix.
  ///
  /// - Requires: `Self` is a square matrix. e.g. `Self.shape[0] == Self.shape[1]`.
  public static var identity: Self {
    precondition(Self.shape[0] == Self.shape[1])
    var r = Self.zero
    for i in 0..<Self.shape[0] {
      r[i, i] = 1
    }
    return r
  }
}

extension FixedSizeMatrix where Rows == Array3<Vector3> {
  /// Creates a matrix with the given scalars, in row-major order.
  @differentiable
  public init(
    _ s00: Double, _ s01: Double, _ s02: Double,
    _ s10: Double, _ s11: Double, _ s12: Double,
    _ s20: Double, _ s21: Double, _ s22: Double
  ) {
    self.init(rows: Vector3(s00, s01, s02), Vector3(s10, s11, s12), Vector3(s20, s21, s22))
  }
}

// MARK: - Conformances to vector space related protocols.

extension FixedSizeMatrix: AdditiveArithmetic {
  public static var zero: Self {
    return Self(rows: Rows((0..<Rows.count).lazy.map { _ in Rows.Element.zero }))
  }
}

extension FixedSizeMatrix: Differentiable {
  public typealias TangentVector = Self
}

extension FixedSizeMatrix: EuclideanVectorN {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
    lhs.withUnsafeMutableBufferPointer { bLhs in
      rhs.withUnsafeBufferPointer { bRhs in
        for i in 0..<Self.dimension {
          bLhs[i] += bRhs[i]
        }
      }
    }
  }

  @derivative(of: +=)
  @usableFromInline
  static func vjpPlusEquals(_ lhs: inout Self, _ rhs: Self) -> (value: (), pullback: (inout Self) -> Self) {
    lhs += rhs
    func pullback(_ v: inout Self) -> Self {
      return v
    }
    return ((), pullback)
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
    lhs.withUnsafeMutableBufferPointer { bLhs in
      rhs.withUnsafeBufferPointer { bRhs in
        for i in 0..<Self.dimension {
          bLhs[i] -= bRhs[i]
        }
      }
    }
  }

  @derivative(of: -=)
  @usableFromInline
  static func vjpMinusEquals(_ lhs: inout Self, _ rhs: Self) -> (value: (), pullback: (inout Self) -> Self) {
    lhs -= rhs
    func pullback(_ v: inout Self) -> Self {
      return -v
    }
    return ((), pullback)
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
    lhs.withUnsafeMutableBufferPointer { b in
      for i in 0..<Self.dimension {
        b[i] *= rhs
      }
    }
  }

  @derivative(of: *=)
  @usableFromInline
  static func vjpTimesEquals(_ lhs: inout Self, _ rhs: Double) -> (value: (), pullback: (inout Self) -> Double) {
    let origLhs = lhs
    lhs *= rhs
    func pullback(_ v: inout Self) -> Double {
      let r = v.dot(origLhs)
      v *= rhs
      return r
    }
    return ((), pullback)
  }

  @differentiable
  public static func /= (_ lhs: inout Self, _ rhs: Double) {
    lhs.withUnsafeMutableBufferPointer { b in
      for i in 0..<Self.dimension {
        b[i] /= rhs
      }
    }
  }

  @derivative(of: /=)
  @usableFromInline
  static func vjpDividedEquals(_ lhs: inout Self, _ rhs: Double) -> (value: (), pullback: (inout Self) -> Double) {
    lhs /= rhs
    let lhsCopy = lhs
    func pullback(_ v: inout Self) -> Double {
      v /= rhs
      return -v.dot(lhsCopy)
    }
    return ((), pullback)
  }

  @differentiable
  public static func / (_ lhs: Self, _ rhs: Double) -> Self {
    var r = lhs
    r /= rhs
    return r
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    withUnsafeBufferPointer { b1 in
      other.withUnsafeBufferPointer { b2 in
        (0..<Self.dimension).reduce(0) { (r, i) in r + b1[i] * b2[i] }
      }
    }
  }

  @derivative(of: dot)
  @usableFromInline
  func vjpDot(_ other: Self) -> (value: Double, pullback: (Double) -> (Self, Self)) {
    return (self.dot(other), { v in (v * other, v * self) })
  }

  public static var dimension: Int {
    return shape[0] * shape[1]
  }

  public static var standardBasis: [Self] {
    (0..<dimension).map { i in
      var v = zero
      v[i] = 1
      return v
    }
  }
}

// MARK: - Matrix math.

extension FixedSizeMatrix {
  public static var isSquare: Bool {
    return shape[0] == shape[1]
  }

  /// Creates the outer product of `lhs` and `rhs`.
  ///
  /// - Requires: `Self.isSquare`.
  public init(outerProduct lhs: Rows.Element, _ rhs: Rows.Element) {
    precondition(Self.isSquare, "init(outerProduct:_:) requires a square matrix")
    self = Self.zero
    for i in 0..<Self.shape[0] {
      for j in 0..<Self.shape[0] {
        self[i, j] = lhs[i] * rhs[j]
      }
    }
  }

  /// Creates the outer product of `lhs` and `rhs`.
  ///
  /// - Requires: `Self.isSquare`.
  public init<Result: EuclideanVectorN>(outerProductNosqr lhs: Result, _ rhs: Rows.Element) {
    self = Self.zero
    for i in 0..<Result.dimension {
      for j in 0..<Self.shape[0] {
        self[i, j] = lhs[i] * rhs[j]
      }
    }
  }

  /// Returns the transpose of `self`.
  ///
  /// - Requires: `Self.isSquare`.
  @differentiable
  public func transposed() -> Self {
    precondition(Self.isSquare, "transposed() requires a square matrix")
    var r = Self.zero
    for i in 0..<Self.shape[0] {
      for j in 0..<Self.shape[0] {
        r[i, j] = self[j, i]
      }
    }
    return r
  }

  @derivative(of: transposed)
  @usableFromInline
  func vjpTransposed() -> (value: Self, pullback: (Self) -> Self) {
    return (transposed(), { $0.transposed() })
  }

  /// Returns the transpose of `self`.
  ///
  /// - Requires: `Self.isSquare`.
  @differentiable
  public func transposed_nonsquare<Result>() -> FixedSizeMatrix<Result> {
    var r = FixedSizeMatrix<Result>.zero
    for i in 0..<FixedSizeMatrix<Result>.shape[0] {
      for j in 0..<FixedSizeMatrix<Result>.shape[1] {
        r[i, j] = self[j, i]
      }
    }
    return r
  }

  @derivative(of: transposed_nonsquare)
  @usableFromInline
  func vjpTransposedNonsquare<Result>() -> (value: FixedSizeMatrix<Result>, pullback: (FixedSizeMatrix<Result>) -> Self) {
    return (transposed_nonsquare(), { $0.transposed_nonsquare() })
  }
}

/// Returns the matrix-vector product of `lhs` and `rhs`.
///
/// Note: This is currently only implemented for square matrices, but could be extended later.
///
/// - Requires: `type(of: lhs).isSquare`.
@differentiable
public func matvec<Rows>(_ lhs: FixedSizeMatrix<Rows>, _ rhs: Rows.Element) -> Rows.Element {
  precondition(type(of: lhs).isSquare, "matvec only implemented for square matrices")
  var r = Rows.Element.zero
  for i in 0..<Rows.Element.dimension {
    for j in 0..<Rows.Element.dimension {
      r[i] += lhs[i, j] * rhs[j]
    }
  }
  return r
}

@derivative(of: matvec)
@usableFromInline
func vjpMatvec<Rows>(_ lhs: FixedSizeMatrix<Rows>, _ rhs: Rows.Element)
  -> (value: Rows.Element, pullback: (Rows.Element) -> (FixedSizeMatrix<Rows>, Rows.Element))
{
  return (
    matvec(lhs, rhs),
    { v in
      (FixedSizeMatrix<Rows>(outerProduct: v, rhs), matvec(lhs.transposed(), v))
    }
  )
}

/// Returns the matrix-vector product of `lhs` and `rhs`.
///
/// Note: This is currently only implemented for square matrices, but could be extended later.
///
/// - Requires: `type(of: lhs).isSquare`.
@differentiable
public func matvec_nonsquare<Rows, Result: EuclideanVectorN>(_ lhs: FixedSizeMatrix<Rows>, _ rhs: Rows.Element) -> Result {
  var r = Result.zero
  for i in 0..<Rows.count {
    for j in 0..<Rows.Element.dimension {
      r[i] += lhs[i, j] * rhs[j]
    }
  }
  return r
}

@derivative(of: matvec_nonsquare)
@usableFromInline
func vjpMatvec_nonsquare<Rows, Result: EuclideanVectorN>(_ lhs: FixedSizeMatrix<Rows>, _ rhs: Rows.Element)
  -> (value: Result, pullback: (Result) -> (FixedSizeMatrix<Rows>, Rows.Element))
{
  return (
    matvec_nonsquare(lhs, rhs),
    { v in
      let t = lhs.transposed_nonsquare()
      return (FixedSizeMatrix<Rows>(outerProductNosqr: v, rhs), matvec_nonsquare(t, v))
    }
  )
}

/// Returns the matrix-matrix product of `lhs` and `rhs`.
///
/// Note: This is currently only implemented for the product of square matrices of identical shape,
/// but could be extended later.
///
/// - Requires: `type(of: lhs).isSquare`.
@differentiable
public func matmul<Rows>(_ lhs: FixedSizeMatrix<Rows>, _ rhs: FixedSizeMatrix<Rows>)
  -> FixedSizeMatrix<Rows>
{
  precondition(type(of: lhs).isSquare, "matmul only implemented for square matrices")
  var r = FixedSizeMatrix<Rows>.zero
  for line in 0..<Rows.Element.dimension {
    for i in 0..<Rows.Element.dimension {
      for j in 0..<Rows.Element.dimension {
        r[i, line] += lhs[i, j] * rhs[j, line]
      }
    }
  }
  return r
}

@derivative(of: matmul)
@usableFromInline
func vjpMatmul<Rows>(_ lhs: FixedSizeMatrix<Rows>, _ rhs: FixedSizeMatrix<Rows>) -> (
  value: FixedSizeMatrix<Rows>,
  pullback: (FixedSizeMatrix<Rows>) -> (FixedSizeMatrix<Rows>, FixedSizeMatrix<Rows>)
) {
  return (
    matmul(lhs, rhs),
    { v in
      (matmul(v, rhs.transposed()), matmul(lhs.transposed(), v))
    }
  )
}

// MARK: - Miscellaneous conformances.

extension FixedSizeMatrix: Equatable {}

extension FixedSizeMatrix: KeyPathIterable {
  public var allKeyPaths: [PartialKeyPath<Self>] {
    (0..<Self.dimension).map { \Self[$0] }
  }
}

extension FixedSizeMatrix: CustomStringConvertible {
  public var description: String {
    "Matrix(" + rows.map { "\($0)" }.joined(separator: ", ") + ")"
  }
}
