import XCTest
@testable import DiffSwiftTest

final class DiffSwiftTestTests: XCTestCase {
  func testBetweenIdentitiesTrivial() {
    let rT1 = Rot2(0), rT2 = Rot2(0);
    let expected = Rot2(0);
    let actual = between(rT1, rT2);

    XCTAssertEqual(actual, expected);
  }

  func testBetweenIdentities() {
    let rT1 = Rot2(0), rT2 = Rot2(2);
    let expected = Rot2(2);
    let actual = between(rT1, rT2);

    XCTAssertEqual(actual, expected);
  }

  func testBetweenDerivatives() {
    let rT1 = Rot2(0), rT2 = Rot2(1);
    print("Initial rT2: ", rT2.theta)
    var model = Between(a: rT1);
    for _ in 0..<10 {
      let (loss, 𝛁loss) = valueWithGradient(at: model) { model -> Double in
        var loss: Double = 0
        let x = Rot2(0);
        let y = rT2;
        let ŷ = model(x)
        let error = between(y, ŷ).theta
        loss = loss + (error * error / 2)

        return loss
      }
      // print("Loss:", loss)
      // print("W: ", 𝛁loss.weight)
      // print("b: ", 𝛁loss.bias)
      model.a.move(along: 𝛁loss.a * 1)
    }

    print("DONE.")
    print("rT1: ", rT1.theta, "rT2: ", rT2.theta)
    print("model: ", model.a.theta, "Error: ", between(rT2, model(Rot2(0))).theta)
  }

  static var allTests = [
    ("testBetweenIdentitiesTrivial", testBetweenIdentitiesTrivial),
    ("testBetweenIdentities", testBetweenIdentities),
    ("testBetweenDerivatives", testBetweenDerivatives)
  ]
}
