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
  var c: Double {
    c_
  }

  @usableFromInline
  @derivative(of: c)
  func _vjpCos() -> (value: Double, pullback: (Double.TangentVector) -> Rot2.TangentVector) {
    (c_, { v in -v * self.s_ })
  }

  @differentiable
  var s: Double {
    s_
  }

  @usableFromInline
  @derivative(of: s)
  func _vjpSin() -> (value: Double, pullback: (Double.TangentVector) -> Rot2.TangentVector) {
    (s_, { v in v * self.c_ })
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
      v
    })
  }

  // Construct from cosine and sine values directly.
  public init(c: Double, s: Double) {
    self.init(atan2wrap(s, c))
  }

  public typealias TangentVector = Double

  public mutating func move(along direction: TangentVector) {
    let r = Rot2(direction) * self
    (c_, s_) = (r.c_, r.s_)
  }

  public var zeroTangentVector: TangentVector {
    Double.zero
  }

  @differentiable
  public var theta: Double {
    atan2wrap(s_, c_)
  }

  @usableFromInline
  @derivative(of: theta)
  func _vjpTheta() -> (value: Double, pullback: (Double.TangentVector) -> Rot2.TangentVector) {
    return (theta, { v in
      v
    })
  }
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

  /// Vector Jacobian Product of product of Rot2
  /// lhs/rhs are the arguments of the function you want to evaluate gradient on
  /// @returns a function that maps from df/dw_n+1 to df/dw_n
  /// vjp stands for Vector Jacobian Product (see below)
  @inlinable
  @derivative(of: *)
  static func _vjpMultiply(lhs: Rot2, rhs: Rot2) -> (value: Rot2, pullback: (Double) -> (Double, Double)) {
    (lhs * rhs, { v in (v, v) })
  }

  // Action on a point
  // Differentiation is automatic as constructor is differentiable and arguments are linear.
  @differentiable
  func rotate(_ p: Point2) -> Point2 {
    Point2(c * p.x + -s * p.y, s * p.x + c * p.y)
  }

  // Action of inverse on a point
  // Differentiation is automatic as constructor is differentiable and arguments are linear.
  @differentiable
  func unrotate(_ p: Point2) -> Point2 {
    Point2(c * p.x + s * p.y, -s * p.x + c * p.y)
  }
}

extension Rot2: CustomDebugStringConvertible {
  public var debugDescription: String {
    "Rot2(theta: \(theta)"
  }
}

// inverse rotation, differentiation is automatic because Rot2 constructor has derivative
@differentiable
public func inverse(_ r: Rot2) -> Rot2 {
  Rot2(c: r.c, s: -r.s)
}

/// Calculate relative rotation between two rotations R1 and R2
@differentiable
public func between(_ R1: Rot2, _ R2: Rot2) -> Rot2 {
  inverse(R1) * R2
}

struct Between: Differentiable {
  var a: Rot2 = Rot2(0)

  @differentiable
  func callAsFunction(_ b: Rot2) -> Rot2 {
    between(a, b)
  }
}

@differentiable
func * (r: Rot2, p: Point2) -> Point2 {
  r.rotate(p)
}
