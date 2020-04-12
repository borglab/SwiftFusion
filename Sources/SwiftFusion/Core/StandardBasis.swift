/// A differentiable type whose `TangentVector` type has an identified "standard" basis.
// TODO(TF-1227): It would be simpler to have a `StandardBasis` protocol, and conform
// `TangentVector` to that, so we should do that after TF-1227 is resolved.
public protocol TangentStandardBasis: Differentiable {
  /// The identified set of "standard" basis vectors.
  static var tangentStandardBasis: [TangentVector] { get }
}

extension Double: TangentStandardBasis {
  public static var tangentStandardBasis: [Double] { [1.0] }
}

/// Provides an implementation of `tangentStandardBasis` so that it is easy to conform types to
/// `TangentStandardBasis`.
///
/// Note: Assumes that scalars are `Double`.
extension Differentiable where TangentVector: KeyPathIterable {
  public static var tangentStandardBasis: [TangentVector] {
    var vectors: [TangentVector] = []
    for kp in TangentVector.zero.recursivelyAllWritableKeyPaths(to: Double.self) {
      var vector = TangentVector.zero
      vector[keyPath: kp] = 1.0
      vectors.append(vector)
    }
    return vectors
  }
}

extension Array where Element: Differentiable {
  public static var tangentStandardBasis: [TangentVector] {
    var vectors: [TangentVector] = []
    for kp in TangentVector.zero.recursivelyAllWritableKeyPaths(to: Double.self) {
      var vector = TangentVector.zero
      vector[keyPath: kp] = 1.0
      vectors.append(vector)
    }
    return vectors
  }
}

extension Array: TangentStandardBasis where Element: Differentiable {
  
}
