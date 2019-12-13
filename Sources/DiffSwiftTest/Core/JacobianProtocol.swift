public protocol JacobianEvaluatable {
  associatedtype ValueType = Double

  var jacobianDim: Int {
    get
  }

  func jacobianAt(n: Int, m: Int) -> ValueType
}

// A jacobian can be calculated by applying the pullback (reverse mode) to the basis vectors of
// the result tangent type, or by applying the differential (forward mode) to the basis vectors of
// the input tangent type.
//
// Swift doesn't yet completely support forward mode, so we'll only do the reverse mode version.

/// Returns the jacobian matrix at the given point, represented as an array of columns.
// Note: We could make this a bit fancier by defining a protocol for "has basis vectors",
//       constraining B.TangentVector to have that protocol, and then automatically getting the
//       basis vectors from that instead of taking them as an argument.
func jacobian<A: Differentiable, B: Differentiable>(
  of f: @differentiable(A) -> B,
  at p: A,
  basisVectors: [B.TangentVector]) -> [A.TangentVector] {
  let pb = pullback(at: p, in: f)
  return basisVectors.map { pb($0) }
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
