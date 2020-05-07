import TensorFlow

/// Generic Lie Group protocol
public protocol LieGroup: Manifold {
  @differentiable(wrt: (lhs, rhs))
  static func * (_ lhs: Self, _ rhs: Self) -> Self
  
  @differentiable
  func inverse() -> Self
  
  @differentiable(wrt: global)
  func local(_ global: Self) -> Self.Coordinate.LocalCoordinate
}
