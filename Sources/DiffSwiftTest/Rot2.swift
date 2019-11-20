import TensorFlow

// I needed to wrap this so that I could provide a derivative, becuase the library does not define
// a derivative for this.
@differentiable(vjp: _vjpAtan2wrap)
func atan2wrap(_ s: Double, _ c: Double) -> Double {
  atan2(s, c)
}

func _vjpAtan2wrap(_ s: Double, _ c: Double) -> (Double, (Double) -> (Double, Double)) {
  let theta = atan2(s, c)
  let normSquared = c * c + s * s
  return (theta, { v in (v * c / normSquared, -v * s / normSquared) })
}

/// Rot2 class is the Swift type for the SO(2) manifold of 2D Rotations around
/// the origin
public struct Rot2: Equatable, Differentiable {
  // TODO: think about the situations where need exact value instead of
  // equivalent classes
  // var theta_ : Double;

  /// Cosine and Sine of the rotation angle
  private var c_, s_: Double

  @differentiable(vjp: _vjpCos)
  var c: Double {
    c_
  }

  @usableFromInline
  func _vjpCos() -> (Double, (Double.TangentVector) -> Rot2.TangentVector) {
    (c_, { v in -v * self.s_ })
  }

  @differentiable(vjp: _vjpSin)
  var s: Double {
    s_
  }

  @usableFromInline
  func _vjpSin() -> (Double, (Double.TangentVector) -> Rot2.TangentVector) {
    (s_, { v in v * self.c_ })
  }

  @differentiable(vjp: _vjpInit)
  public init(_ theta: Double) {
    // theta_ = theta;
    c_ = cos(theta)
    s_ = sin(theta)
  }

  @usableFromInline
  static func _vjpInit(_ theta: Double) -> (Self, (TangentVector) -> Double) {
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

  @differentiable(vjp: _vjpTheta)
  public var theta: Double {
    atan2wrap(s_, c_)
  }

  @usableFromInline
  func _vjpTheta() -> (Double, (Double.TangentVector) -> Rot2.TangentVector) {
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

  @differentiable(vjp: _vjpMultiply(lhs:rhs:))
  static func * (lhs: Rot2, rhs: Rot2) -> Rot2 {
    Rot2(
      c: lhs.c * rhs.c - lhs.s * rhs.s,
      s: lhs.s * rhs.c + lhs.c * rhs.s)
  }

  @inlinable
  static func _vjpMultiply(lhs: Rot2, rhs: Rot2) -> (Rot2, (Double) -> (Double, Double)) {
    return (lhs * rhs, { v in
      (v, v)
    })
  }

  @differentiable
  func unrotate(_ p: Point2) -> Point2 {
    return Point2(c * p.x + s * p.y, -s * p.x + c * p.y)
  }
}

extension Rot2: CustomDebugStringConvertible {
    public var debugDescription: String {
        return "Rot2(theta: \(theta)"
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
  return Point2(r.c * p.x + -r.s * p.y, r.s * p.x + r.c * p.y);
}