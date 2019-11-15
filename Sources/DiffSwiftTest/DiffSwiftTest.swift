import TensorFlow
struct DiffSwiftTest {
    var text = "Hello, World!"
}

/// Rot2 class is the Swift type for the SO(2) manifold of 2D Rotations around
/// the origin
public struct Rot2 : Equatable, Differentiable {
  // TODO: think about the situations where need exact value instead of
  // equivalent classes
  // var theta_ : Double;
  
  /// Cosine and Sine of the rotation angle
  var c_, s_ : Double;

  @differentiable(vjp: _vjpInit)
  public init(_ theta: Double) {
    // theta_ = theta;
    c_ = cos(theta);
    s_ = sin(theta);
  }

  @usableFromInline
  static func _vjpInit(_ theta: Double) -> (Self, (TangentVector) -> Double) {
    return (.init(theta), { v in
      theta
    })
  }

  public init(c: Double, s: Double) {
    (c_, s_) = (c, s);
  }

  public typealias TangentVector = Double;

  public mutating func move(along direction: TangentVector) {
    let r = Rot2(direction) * self;
    (c_, s_) = (r.c_, r.s_)
  }

  public var zeroTangentVector: TangentVector {
    Double.zero
  }
  
  @differentiable(vjp: _vjpTheta)
  public var theta: Double {
    return atan2(c_, s_)
  }

  @usableFromInline
  func _vjpTheta() -> (Double , (Double.TangentVector) -> Rot2.TangentVector) {
    return (atan2(self.c_, self.s_), { v in
      atan2(self.c_, self.s_)
    })
  }
}

infix operator *: MultiplicationPrecedence
public extension Rot2 {
  static func == (lhs: Rot2, rhs: Rot2) -> Bool {
    return (lhs.c_, lhs.s_) == (rhs.c_, rhs.s_);
  }

  @differentiable(vjp: _vjpMultiply(lhs:rhs:))
  static func * (lhs: Rot2, rhs: Rot2) -> Rot2 {
    return Rot2(
      c: lhs.c_ * rhs.c_ - lhs.s_ * rhs.s_,
      s: lhs.s_ * rhs.c_ + lhs.c_ * rhs.s_
    );
  }

  @inlinable
  static func _vjpMultiply(lhs: Rot2, rhs: Rot2) -> (Rot2, (Double) -> (Double, Double)) {
      return (lhs * rhs, { v in
          let lhsGrad = v
          let rhsGrad = v
          return (lhsGrad, rhsGrad)
      })
  }
}

@differentiable(vjp: _vjpInverse(r:))
public func inverse(_ r: Rot2) -> Rot2 {
  return Rot2(c: r.c_, s: -r.s_);
}

@inlinable
func _vjpInverse(r: Rot2) -> (Rot2, (Double) -> (Double)) {
    return (inverse(r), { v in
        let rGrad = -v
        return (rGrad)
    })
}

/// Calculate relative rotation between two rotations rT1 and rT2
@differentiable
public func between(_ rT1:Rot2, _ rT2:Rot2) -> Rot2 {
  return inverse(rT1) * rT2;
}

struct Between: Differentiable {
    var a: Rot2 = Rot2(0);

    @differentiable
    func callAsFunction(_ b: Rot2) -> Rot2 {
        between(a, b)
    }
}

/// Class testing SO(2) manifold type Rot2
class TestRot2 {
  func testBetweenIdentitiesTrivial() {
    let rT1 = Rot2(0), rT2 = Rot2(0);
    let expected = Rot2(0);
    let actual = between(rT1, rT2);

    assert(actual == expected);
  }
  func testBetweenIdentities() {
    let rT1 = Rot2(0), rT2 = Rot2(2);
    let expected = Rot2(2);
    let actual = between(rT1, rT2);

    assert(actual == expected);
  }

  func testBetweenDerivatives() {
    let rT1 = Rot2(0), rT2 = Rot2(1);
    var model = Between(a: rT1);
    for _ in 0..<400 {
      let (loss, 𝛁loss) = valueWithGradient(at: model) { model -> Double in
        var loss: Double = 0
        let x = Rot2(0);
        let y = rT2;
        let ŷ = model(x)
        let error = between(y, ŷ).theta
        loss = loss + (error * error / 2)

        return loss
      }
      print("Loss:", loss)
      // print("W: ", 𝛁loss.weight)
      // print("b: ", 𝛁loss.bias)
      model.a.move(along: 𝛁loss.a * 0.01)
    }

    print("DONE.")
    print("rT1: ", rT1.theta, "rT2: ", rT2.theta)
    print("model: ", model.a.theta, "Error: ", between(rT2, model(Rot2(0))).theta)
  }
}
