import Foundation
import XCTest

import SwiftFusion

class DictionaryDifferentiableTests: XCTestCase {
  /// Test the `AdditiveArithmetic` `zero` requirement.
  func testAdditiveArithmeticZero() {
    XCTAssertEqual(Dictionary<String, Float>.zero, [:])
  }

  /// Test the `AdditiveArithmetic` `+` requirement.
  func testAdditiveArithmeticPlus() {
    XCTAssertEqual(
      ([:] as [String: Int]) + [:],
      [:]
    )
    XCTAssertEqual(
      ["a": 1] + [:],
      ["a": 1]
    )
    XCTAssertEqual(
      [:] + ["b": 1],
      ["b": 1]
    )
    XCTAssertEqual(
      ["a": 1] + ["b": 1],
      ["a": 1, "b": 1]
    )
    XCTAssertEqual(
      ["a": 1, "b": 1] + ["b": 1],
      ["a": 1, "b": 2]
    )
  }

  /// Test the `AdditiveArithmetic` `-` requirement.
  func testAdditiveArithmeticMinus() {
    XCTAssertEqual(
      ([:] as [String: Int]) - [:],
      [:]
    )
    XCTAssertEqual(
      ["a": 1] - [:],
      ["a": 1]
    )
    XCTAssertEqual(
      [:] - ["b": 1],
      ["b": -1]
    )
    XCTAssertEqual(
      ["a": 1] - ["b": 1],
      ["a": 1, "b": -1]
    )
    XCTAssertEqual(
      ["a": 1, "b": 1] - ["b": 1],
      ["a": 1, "b": 0]
    )
  }

  /// Test the `Differentiable` `move` requirement.
  func testMove() {
    var foo: [String: Double] = ["a": 0, "b": 0]
    foo.move(along: ["a": 1])
    XCTAssertEqual(foo, ["a": 1, "b": 0])
    foo.move(along: ["b": 1])
    XCTAssertEqual(foo, ["a": 1, "b": 1])
    foo.move(along: ["a": 1, "b": 2])
    XCTAssertEqual(foo, ["a": 2, "b": 3])
  }

  /// Test the `Differentiable` `zeroTangentVector` requirement.
  func testZeroTangentVector() {
    XCTAssertEqual(
      ([:] as [String: Double]).zeroTangentVector,
      [:]
    )
    XCTAssertEqual(
      (["a": 1] as [String: Double]).zeroTangentVector,
      ["a":  0]
    )
  }

  /// Test the value and derivative of `differentiableSubscript`.
  func testDifferentiableSubscript() {
    let point: [String: Double] = ["a": 1, "b": 2]
    XCTAssertEqual(point.differentiableSubscript("a"), 1)
    XCTAssertEqual(point.differentiableSubscript("b"), 2)
    XCTAssertEqual(
      gradient(at: point) { $0.differentiableSubscript("a") },
      ["a": 1]
    )
    XCTAssertEqual(
      gradient(at: point) { $0.differentiableSubscript("a") + $0.differentiableSubscript("b") },
      ["a": 1, "b": 1]
    )
  }

  static var allTests = [
    ("testAdditiveArithmeticZero", testAdditiveArithmeticZero),
    ("testAdditiveArithmeticPlus", testAdditiveArithmeticPlus),
    ("testAdditiveArithmeticMinus", testAdditiveArithmeticMinus),
    ("testMove", testMove),
    ("testZeroTangentVector", testZeroTangentVector),
    ("testDifferentiableSubscript", testDifferentiableSubscript)
  ]
}
