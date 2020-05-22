import TensorFlow

/// Rot2 class is the Swift type for the SO(2) manifold of 2D Rotations around
/// the origin.
public struct Rot2: Manifold, LieGroup, Equatable, KeyPathIterable {

  // MARK: - Manifold conformance

  public typealias Coordinate = Rot2Coordinate
  public typealias TangentVector = Vector1

  public var coordinateStorage: Rot2Coordinate
  public init(coordinateStorage: Rot2Coordinate) { self.coordinateStorage = coordinateStorage }

  public mutating func move(along direction: Coordinate.LocalCoordinate) {
    coordinateStorage = coordinateStorage.retract(direction)
  }

  // MARK: - Convenience initializers and computed properties

  // Construct from theta.
  @differentiable
  public init(_ theta: Double) {
    self.init(coordinate: Rot2Coordinate(c: cos(theta), s: sin(theta)))
  }

  @differentiable
  public var theta: Double { coordinate.theta }

  /// Construct from cosine and sine values directly.
  @differentiable
  public init(c: Double, s: Double) {
    self.init(atan2wrap(s, c))
  }

  /// Cosine value.
  @differentiable
  public var c: Double { coordinate.c }

  /// Sine value.
  @differentiable
  public var s: Double { coordinate.s }
}

extension Rot2: CustomDebugStringConvertible {
  public var debugDescription: String {
    "Rot2(theta: \(theta))"
  }
}

extension Rot2 {
  /// Returns the result of acting `self` on `v`.
  @differentiable
  func rotate(_ v: Vector2) -> Vector2 {
    Vector2(c * v.x - s * v.y, s * v.x + c * v.y)
  }

  /// Returns the result of acting the inverse of `self` on `v`.
  @differentiable
  func unrotate(_ v: Vector2) -> Vector2 {
    Vector2(c * v.x + s * v.y, -s * v.x + c * v.y)
  }
}

@differentiable
func *^ (r: Rot2, p: Vector2) -> Vector2 {
  r.rotate(p)
}

// MARK: - Global coordinate system

public struct Rot2Coordinate: Equatable, KeyPathIterable {
  public var c, s: Double
}

public extension Rot2Coordinate {
  @differentiable
  init(_ theta: Double) {
    self.c = cos(theta)
    self.s = sin(theta)
  }

  @differentiable
  var theta: Double {
    atan2wrap(s, c)
  }
}

extension Rot2Coordinate: LieGroupCoordinate {
  /// Creates the group identity.
  public init() {
    self.init(0)
  }

  /// Product of two rotations.
  @differentiable
  public static func ** (lhs: Rot2Coordinate, rhs: Rot2Coordinate) -> Rot2Coordinate {
    Rot2Coordinate(
      c: lhs.c * rhs.c - lhs.s * rhs.s,
      s: lhs.s * rhs.c + lhs.c * rhs.s)
  }

  /// Inverse of the rotation.
  @differentiable
  public func inverse() -> Rot2Coordinate {
    Rot2Coordinate(c: self.c, s: -self.s)
  }
}

extension Rot2Coordinate: ManifoldCoordinate {
  public typealias LocalCoordinate = Vector1

  @differentiable(wrt: local)
  public func retract(_ local: Vector1) -> Self {
    self ** Rot2Coordinate(local.x)
  }

  @differentiable(wrt: global)
  public func localCoordinate(_ global: Self) -> Vector1 {
    Vector1((self.inverse() ** global).theta)
  }
}

// MARK: - Helper functions

// We need a special version of atan2 that provides a derivative.
@differentiable
fileprivate func atan2wrap(_ s: Double, _ c: Double) -> Double {
  atan2(s, c)
}

// Implement derivative of atan2wrap.
// d atan2(s,c)/s = c / (s^2+c^2)
// d atan2(s,c)/s = -s / (s^2+c^2)
// TODO(frank): make use of fact that s^2 + c^2 = 1
@derivative(of: atan2wrap)
fileprivate func _vjpAtan2wrap(_ s: Double, _ c: Double) -> (value: Double, pullback: (Double) -> (Double, Double)) {
  let theta = atan2(s, c)
  let normSquared = c * c + s * s
  return (theta, { v in (v * c / normSquared, -v * s / normSquared) })
}
