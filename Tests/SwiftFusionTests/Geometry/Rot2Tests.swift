// This file tests for the various identities in Rot2

@testable import SwiftFusion
// @testable import SwiftFusion.Geometry.Rot2

import XCTest

// Boilerplate code for running the test cases
import XCTest

final class Rot2Tests: XCTestCase {

  // checks between function for two identity rotations
  func testBetweenIdentitiesTrivial() {
    let R1 = Rot2(0), R2 = Rot2(0)
    let expected = Rot2(0)
    let actual = between(R1, R2)

    XCTAssertEqual(actual, expected)
  }

  // checks between
  func testBetween() {
    let R1 = Rot2(0), R2 = Rot2(2)
    let expected = Rot2(2)
    let actual = between(R1, R2)

    XCTAssertEqual(actual, expected)
  }

  // Check derivatives for between
  func testBetweenDerivatives() {
    let R1 = Rot2(0), R2 = Rot2(2)
    let (_, 𝛁actual1) = valueWithGradient(at: R1) { R -> Double in
      between(R, R2).theta
    }

    let (_, 𝛁actual2) = valueWithGradient(at: R2) { R -> Double in
      between(R1, R).theta
    }

    XCTAssertEqual(𝛁actual1, -1.0)
    XCTAssertEqual(𝛁actual2, 1.0)
  }

  // Check gradient descent
  func testGradientDescent() {
    var R1 = Rot2(0), R2 = Rot2(1)
    print("Initial R2: ", R2.theta)

    for _ in 0..<100 {
      var (_, 𝛁loss) = valueWithGradient(at: R1) { R1 -> Double in
        var loss: Double = 0
        let ŷ = between(R1, R2)
        let error = ŷ.theta
        loss = loss + (error * error / 10)

        return loss
      }

      // print("𝛁loss", 𝛁loss)
      𝛁loss = -𝛁loss
      R1.move(along: 𝛁loss)
    }

    print("DONE.")
    print("R1: ", R1.theta, "R2: ", R2.theta)

    XCTAssertEqual(R1.theta, R2.theta, accuracy: 1e-5)
  }

  static var allTests = [
    ("testBetweenIdentitiesTrivial", testBetweenIdentitiesTrivial),
//    ("testBetweenIdentities", testBetweenIdentities),
    ("testBetweenDerivatives", testBetweenDerivatives),
//    ("testRot2Derivatives", testRot2Derivatives),
  ]
}
