public protocol JacobianEvaluatable: Differentiable, KeyPathIterable where TangentVector: KeyPathIterable {
  
}

// A jacobian can be calculated by applying the pullback (reverse mode) to the basis vectors of
// the result tangent type, or by applying the differential (forward mode) to the basis vectors of
// the input tangent type.
//
// Swift doesn't yet completely support forward mode, so we'll only do the reverse mode version.

/// Returns the jacobian matrix at the given point, represented as an array of columns.
/// Note: We could make this a bit fancier by defining a protocol for "has basis vectors",
///      constraining B.TangentVector to have that protocol, and then automatically getting the
///      basis vectors from that instead of taking them as an argument.
///
/// Additional Notes
/// ===================
/// For example, if we have an array of `Point2` `map = [p1, p2, p3]` with an operation of `map[1] - map[0]`,
/// the result will be
/// ```
/// J(ef) = [
///  [ [-1.0, 0.0],
///    [1.0, 0.0],
///    [0.0, 0.0] ]
///  [ [0.0, -1.0],
///    [0.0, 1.0],
///    [0.0, 0.0] ]
///  ]
///  ```
/// So this is 2x3 but the data type is Point2.TangentVector.
/// In "normal" Jacobian notation, we should have a 2x6.
/// ```
/// [ [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
///   [0.0, -1.0, 0.0, 1.0, 0.0, 0.0] ]
/// ```
func jacobian<A: Differentiable, B: JacobianEvaluatable>(
  of f: @differentiable(A) -> B,
  at p: A,
  basisVectors: [B.TangentVector] = B.TangentVector.basisVectors()) -> [A.TangentVector] {
  let pb = pullback(at: p, in: f)
  return basisVectors.map { pb($0) }
}


func jacobian<A: Differentiable>(
  of f: @differentiable(A) -> Double,
  at p: A) -> [A.TangentVector] {
  let pb = pullback(at: p, in: f)
  return [1.0].map { pb($0) }
}

func jacobian<A: Differentiable>(
  of f: @differentiable(A) -> Double,
  at p: A,
  basisVectors: [Double]) -> [A.TangentVector] {
  let pb = pullback(at: p, in: f)
  return basisVectors.map { pb($0) }
}

public extension KeyPathIterable where Self: AdditiveArithmetic {
  // Note: Assumes that the scalars are Double.
  static func basisVectors() -> [Self] {
    var vectors: [Self] = []
    for kp in zero.recursivelyAllWritableKeyPaths(to: Double.self) {
      var vector = zero
      vector[keyPath: kp] = 1.0
      vectors.append(vector)
    }
    return vectors
  }
}
