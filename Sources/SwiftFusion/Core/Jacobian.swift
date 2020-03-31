// A jacobian can be calculated by applying the pullback (reverse mode) to the basis vectors of
// the result tangent type, or by applying the differential (forward mode) to the basis vectors of
// the input tangent type.
//
// Swift doesn't yet completely support forward mode, so we'll only do the reverse mode version.

/// Returns the jacobian matrix at the given point, represented as an array of columns.
///
/// Additional Notes
/// ===================
/// For example, if we have an array of `Point2` `pts = [p1, p2, p3]` with an operation of `pts[1] - pts[0]`,
/// the result should be
/// ```
/// J_f(pts) = [
///  [ [-1.0, 0.0],
///    [1.0, 0.0],
///    [0.0, 0.0] ]
///  [ [0.0, -1.0],
///    [0.0, 1.0],
///    [0.0, 0.0] ]
///  ]
///  ```
/// So this is 2x3 but the data type is Point2.TangentVector.
/// In "flattened" Jacobian notation, we should have a 2x6.
/// ```
/// [ [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
///   [0.0, -1.0, 0.0, 1.0, 0.0, 0.0] ]
/// ```
public func jacobian<A: Differentiable, B: TangentStandardBasis>(
  of f: @differentiable(A) -> B,
  at p: A,
  basisVectors: [B.TangentVector] = B.tangentStandardBasis
) -> [A.TangentVector] {
  let pb = pullback(at: p, in: f)
  return basisVectors.map { pb($0) }
}
