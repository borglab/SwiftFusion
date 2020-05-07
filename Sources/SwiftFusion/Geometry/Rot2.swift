import TensorFlow

// We need a special version of atan2 that provides a derivative.
@differentiable
func atan2wrap(_ s: Double, _ c: Double) -> Double {
  atan2(s, c)
}

// Implement derivative of atan2wrap.
// d atan2(s,c)/s = c / (s^2+c^2)
// d atan2(s,c)/s = -s / (s^2+c^2)
// TODO(frank): make use of fact that s^2 + c^2 = 1
@derivative(of: atan2wrap)
func _vjpAtan2wrap(_ s: Double, _ c: Double) -> (value: Double, pullback: (Double) -> (Double, Double)) {
  let theta = atan2(s, c)
  let normSquared = c * c + s * s
  return (theta, { v in (v * c / normSquared, -v * s / normSquared) })
}

// @derivative(of: bar)
// public func _(_ x: Float) -> (value: Float, differential: (Float) -> Float) {
//   (x, { dx in dx })
// }

/// Rot2 class is the Swift type for the SO(2) manifold of 2D Rotations around
/// the origin.
public struct Rot2: Equatable, Differentiable, KeyPathIterable {
  // TODO: think about the situations where need exact value instead of
  // equivalent classes
  // var theta_ : Double;

  /// Cosine and Sine of the rotation angle
  private var c_, s_: Double

  @differentiable
  public var c: Double {
    c_
  }

  @usableFromInline
  @derivative(of: c)
  func _vjpCos() -> (value: Double, pullback: (Double) -> TangentVector) {
    (c_, { v in Vector1(-v * self.s_) })
  }

  @differentiable
  public var s: Double {
    s_
  }

  @usableFromInline
  @derivative(of: s)
  func _vjpSin() -> (value: Double, pullback: (Double) -> TangentVector) {
    (s_, { v in Vector1(v * self.c_) })
  }

  // Construct from angle theta.
  @differentiable
  public init(_ theta: Double) {
    c_ = cos(theta)
    s_ = sin(theta)
  }

  @usableFromInline
  @derivative(of: init)
  static func _vjpInit(_ theta: Double) -> (value: Self, pullback: (TangentVector) -> Double) {
    return (Rot2(theta), { v in
      v.x
    })
  }

  // Construct from cosine and sine values directly.
  public init(c: Double, s: Double) {
    self.init(atan2wrap(s, c))
  }

  public typealias TangentVector = Vector1

  public mutating func move(along direction: TangentVector) {
    let r = Rot2(direction.x) * self
    (c_, s_) = (r.c_, r.s_)
  }

  public var zeroTangentVector: TangentVector {
    TangentVector.zero
  }

  @differentiable
  public var theta: Double {
    atan2wrap(s_, c_)
  }

  @usableFromInline
  @derivative(of: theta)
  func _vjpTheta() -> (value: Double, pullback: (Double) -> TangentVector) {
    return (theta, { v in
      Vector1(v)
    })
  }
}

extension Rot2: TangentStandardBasis {
  public static var tangentStandardBasis: [Vector1] { [Vector1(1)] }
}

infix operator *: MultiplicationPrecedence
public extension Rot2 {
  static func == (lhs: Rot2, rhs: Rot2) -> Bool {
    (lhs.c, lhs.s) == (rhs.c, rhs.s)
  }

  /// This is the product of two 2D rotations.
  /// @differentiable is an attribute marker of differentiablity.
  @differentiable
  static func * (lhs: Rot2, rhs: Rot2) -> Rot2 {
    Rot2(
      c: lhs.c * rhs.c - lhs.s * rhs.s,
      s: lhs.s * rhs.c + lhs.c * rhs.s)
  }

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
  
  // inverse rotation, differentiation is automatic because Rot2 constructor has derivative
  @differentiable
  func inverse() -> Rot2 {
    Rot2(c: self.c, s: -self.s)
  }
}

extension Rot2: CustomDebugStringConvertible {
  public var debugDescription: String {
    "Rot2(theta: \(theta))"
  }
}

/// Calculate relative rotation between two rotations R1 and R2
@differentiable
public func between(_ R1: Rot2, _ R2: Rot2) -> Rot2 {
  R1.inverse() * R2
}

@differentiable
func * (r: Rot2, p: Vector2) -> Vector2 {
  r.rotate(p)
}
