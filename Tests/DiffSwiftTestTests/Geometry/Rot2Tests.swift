// This file tests for the various identities in Rot2

@testable import DiffSwiftTest
// @testable import DiffSwiftTest.Geometry.Rot2

import XCTest

// Boilerplate code for running the test cases
import XCTest

final class Rot2Tests: XCTestCase {
  func testBetweenIdentitiesTrivial() {
    let rT1 = Rot2(0), rT2 = Rot2(0)
    let expected = Rot2(0)
    let actual = between(rT1, rT2)

    XCTAssertEqual(actual, expected)
  }

  func testBetweenIdentities() {
    let rT1 = Rot2(0), rT2 = Rot2(2)
    let expected = Rot2(2)
    let actual = between(rT1, rT2)

    XCTAssertEqual(actual, expected)
  }

  func testRot2Derivatives() {
    let rT1 = Rot2(0), rT2 = Rot2(2)
    let (_, 𝛁actual1) = valueWithGradient(at: rT1) { rT1 -> Double in
      between(rT1, rT2).theta
    }

    let (_, 𝛁actual2) = valueWithGradient(at: rT2) { rT2 -> Double in
      between(rT1, rT2).theta
    }

    XCTAssertEqual(𝛁actual1, -1.0)
    XCTAssertEqual(𝛁actual2, 1.0)
  }

  func testBetweenDerivatives() {
    var rT1 = Rot2(0), rT2 = Rot2(1)
    print("Initial rT2: ", rT2.theta)

    for _ in 0..<100 {
      var (_, 𝛁loss) = valueWithGradient(at: rT1) { rT1 -> Double in
        var loss: Double = 0
        let ŷ = between(rT1, rT2)
        let error = ŷ.theta
        loss = loss + (error * error / 10)

        return loss
      }

      // print("𝛁loss", 𝛁loss)
      𝛁loss = -𝛁loss
      rT1.move(along: 𝛁loss)
    }

    print("DONE.")
    print("rT1: ", rT1.theta, "rT2: ", rT2.theta)

    XCTAssertEqual(rT1.theta, rT2.theta, accuracy: 1e-5)
  }

  static var allTests = [
    ("testBetweenIdentitiesTrivial", testBetweenIdentitiesTrivial),
    ("testBetweenIdentities", testBetweenIdentities),
    ("testBetweenDerivatives", testBetweenDerivatives),
    ("testRot2Derivatives", testRot2Derivatives),
  ]
}
