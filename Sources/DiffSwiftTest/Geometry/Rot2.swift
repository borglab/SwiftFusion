import TensorFlow

// I needed to wrap this so that I could provide a derivative, becuase the library does not define
// a derivative for this.
@differentiable
func atan2wrap(_ s: Double, _ c: Double) -> Double {
  atan2(s, c)
}

// @derivative(of: bar)
// public func _(_ x: Float) -> (value: Float, differential: (Float) -> Float) {
//   (x, { dx in dx })
// }

@derivative(of: atan2wrap)
func _vjpAtan2wrap(_ s: Double, _ c: Double) -> (value: Double, pullback: (Double) -> (Double, Double)) {
  let theta = atan2(s, c)
  let normSquared = c * c + s * s
  return (theta, { v in (v * c / normSquared, -v * s / normSquared) })
}

/// Rot2 class is the Swift type for the SO(2) manifold of 2D Rotations around
/// the origin
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

  @differentiable
  public init(_ theta: Double) {
    // theta_ = theta;
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

  /// This is the product of two 2D rotations
  /// @differentiable is an attribute marker of differntiablity
  /// vjp stands for Vector Jacobian Product (see below)
  @differentiable
  static func * (lhs: Rot2, rhs: Rot2) -> Rot2 {
    Rot2(
      c: lhs.c * rhs.c - lhs.s * rhs.s,
      s: lhs.s * rhs.c + lhs.c * rhs.s)
  }

  /// Vector Jacobian Product of product of Rot2
  /// lhs/rhs are the arguments of the function you want to evaluate gradient on
  /// @returns a function that maps from df/dw_n+1 to df/dw_n
  @inlinable
  @derivative(of: *)
  static func _vjpMultiply(lhs: Rot2, rhs: Rot2) -> (value: Rot2, pullback: (Double) -> (Double, Double)) {
    return (lhs * rhs, { v in
      (v, v)
    })
  }

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

@differentiable
public func inverse(_ r: Rot2) -> Rot2 {
  Rot2(c: r.c, s: -r.s)
}

/// Calculate relative rotation between two rotations rT1 and rT2
@differentiable
public func between(_ rT1: Rot2, _ rT2: Rot2) -> Rot2 {
  inverse(rT1) * rT2
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
  Point2(r.c * p.x + -r.s * p.y, r.s * p.x + r.c * p.y)
}
