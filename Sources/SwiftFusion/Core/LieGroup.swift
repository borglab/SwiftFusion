import TensorFlow

/// Generic Lie Group protocol
public protocol LieGroup: Manifold {
  @differentiable(wrt: (lhs, rhs))
  static func * (_ lhs: Self, _ rhs: Self) -> Self
  
  @differentiable
  func inverse() -> Self
  
  @differentiable(wrt: global)
  func localCoordinate(_ global: Self) -> Self.Coordinate.LocalCoordinate
}

/// Calculate relative pose 1T2 between two poses wT1 and wT2
@differentiable(wrt: (wT1, wT2))
public func between<T: LieGroup & Differentiable>(_ wT1: T, _ wT2: T) -> T {
  wT1.inverse() * wT2
}
