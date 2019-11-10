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

  @differentiable
  init(theta: Double) {
    // theta_ = theta;
    (c_, s_) = (cos(theta), sin(theta));
  }

  init(c: Double, s: Double) {
    (c_, s_) = (c, s);
  }

  public typealias TangentVector = Double;

  public mutating func move(along direction: TangentVector) {
    let r = Rot2(theta: direction) * self;
    (c_, s_) = (r.c_, r.s_)
  }

  public var zeroTangentVector: TangentVector {
    Double.zero
  }
}

infix operator *: MultiplicationPrecedence
public extension Rot2 {
  static func == (lhs: Rot2, rhs: Rot2) -> Bool {
    return (lhs.c_, lhs.s_) == (rhs.c_, rhs.s_);
  }

  static func * (lhs: Rot2, rhs: Rot2) -> Rot2 {
    return Rot2(
      c: lhs.c_ * rhs.c_ - lhs.s_ * rhs.s_,
      s: lhs.s_ * rhs.c_ + lhs.c_ * rhs.s_
    );
  }
}

public func inverse(_ r: Rot2) -> Rot2 {
  return Rot2(c: r.c_, s: -r.s_);
}

/// Calculate relative rotation between two rotations rT1 and rT2
@differentiable
public func between(_ rT1:Rot2, _ rT2:Rot2) -> Rot2 {
  return inverse(rT1) * rT2;
}

struct Between: Differentiable {
    var a: Rot2 = Rot2(theta: 0);

    @differentiable
    func callAsFunction(_ b: Rot2) -> Rot2 {
        between(a, b)
    }
}

/// Class testing SO(2) manifold type Rot2
class TestRot2 {
  func testBetweenIdentitiesTrivial() {
    let rT1 = Rot2(theta: 0), rT2 = Rot2(theta: 0);
    let expected = Rot2(theta: 0);
    let actual = between(rT1, rT2);

    assert(actual == expected);
  }
  func testBetweenIdentities() {
    let rT1 = Rot2(theta: 0), rT2 = Rot2(theta: 2);
    let expected = Rot2(theta: 2);
    let actual = between(rT1, rT2);

    assert(actual == expected);
  }

  func testBetweenDerivatives() {
    let rT1 = Rot2(theta: 0), rT2 = Rot2(theta: 2);
    var model = Between(a: rT1);
    for _ in 0..<200 {
      let (loss, 𝛁loss) = valueWithGradient(at: model) { model -> Float in
        var loss: Float = 0
        let x = Rot2(theta: 0);
        let y = rT2;
        let ŷ = model(x)
        let error = atan2(between(y, ŷ).c_, between(y, ŷ).s_)
        loss = loss + Float(error * error / 2)

        return loss
      }
      print("Loss:", loss)
      // print("W: ", 𝛁loss.weight)
      // print("b: ", 𝛁loss.bias)
      model.a.move(along: 𝛁loss.a * 0.01)
    }

    print("DONE.")
  }
}
