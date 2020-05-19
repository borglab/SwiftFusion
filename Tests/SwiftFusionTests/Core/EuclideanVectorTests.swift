import Foundation
import XCTest

import SwiftFusion

/// Conform to this and call `runAllTests()` to test all the `EuclideanVector` requirements on a
/// concrete type.
protocol EuclideanVectorTests {
  /// The concrete type that we are testing.
  associatedtype T: EuclideanVector

  /// The dimension of the vector we are testing.
  var dimension: Int { get }

  /// A set of basis vectors.
  var basisVectors: [T] { get }

  /// Make a vector whose first element is `start` and whose subsequent elements increment by
  /// `stride`.
  func makeVector(from start: Double, stride: Double) -> T
}

extension EuclideanVectorTests {
  /// Runs all the tests.
  func runAllTests() {
    testAddMutating()
    testAddFunctional()
    testSubtractMutating()
    testSubtractFunctional()
    testScalarProductMutating()
    testScalarProductFunctional()
    testDot()
    testNegate()
    testSquaredNorm()
    testNorm()
  }

  /// Tests the value and derivative of the mutating +=.
  func testAddMutating() {
    func doMutatingAdd(_ v1: T, _ v2: T) -> T {
      var result = v1
      result += v2
      return result
    }

    let v1 = makeVector(from: 1, stride: 1)
    let v2 = makeVector(from: 2, stride: 1)
    XCTAssertEqual(doMutatingAdd(v1, v2), makeVector(from: 3, stride: 2))

    let (value, pb) = valueWithPullback(at: v1, v2, in: doMutatingAdd)
    XCTAssertEqual(value, makeVector(from: 3, stride: 2))
    for v in basisVectors {
      XCTAssertEqual(pb(v).0, v)
      XCTAssertEqual(pb(v).1, v)
    }
  }

  /// Tests the value and derivative of the functional +.
  func testAddFunctional() {
    let v1 = makeVector(from: 1, stride: 1)
    let v2 = makeVector(from: 2, stride: 1)
    XCTAssertEqual(v1 + v2, makeVector(from: 3, stride: 2))

    let (value, pb) = valueWithPullback(at: v1, v2, in: +)
    XCTAssertEqual(value, makeVector(from: 3, stride: 2))
    for v in basisVectors {
      XCTAssertEqual(pb(v).0, v)
      XCTAssertEqual(pb(v).1, v)
    }
  }

  /// Tests the value and derivative of the mutating -=.
  func testSubtractMutating() {
    func doMutatingSubtract(_ v1: T, _ v2: T) -> T {
      var result = v1
      result -= v2
      return result
    }

    let v1 = makeVector(from: 1, stride: 1)
    let v2 = makeVector(from: 1, stride: -1)
    XCTAssertEqual(doMutatingSubtract(v1, v2), makeVector(from: 0, stride: 2))

    let (value, pb) = valueWithPullback(at: v1, v2, in: doMutatingSubtract)
    XCTAssertEqual(value, makeVector(from: 0, stride: 2))
    for v in basisVectors {
      XCTAssertEqual(pb(v).0, v)
      XCTAssertEqual(pb(v).1, -v)
    }
  }

  /// Tests the value and derivative of the functional -.
  func testSubtractFunctional() {
    let v1 = makeVector(from: 1, stride: 1)
    let v2 = makeVector(from: 1, stride: -1)
    XCTAssertEqual(v1 - v2, makeVector(from: 0, stride: 2))

    let (value, pb) = valueWithPullback(at: v1, v2, in: -)
    XCTAssertEqual(value, makeVector(from: 0, stride: 2))
    for v in basisVectors {
      XCTAssertEqual(pb(v).0, v)
      XCTAssertEqual(pb(v).1, -v)
    }
  }

  /// Tests the value and derivative of the mutating *=.
  func testScalarProductMutating() {
    func doMutatingScalarProduct(_ s: Double, _ v1: T) -> T {
      var result = v1
      result *= s
      return result
    }

    let s = Double(2)
    let v1 = makeVector(from: 1, stride: 1)
    XCTAssertEqual(doMutatingScalarProduct(s, v1), makeVector(from: 2, stride: 2))

    let (value, pb) = valueWithPullback(at: s, v1, in: doMutatingScalarProduct)
    XCTAssertEqual(value, makeVector(from: 2, stride: 2))
    for (index, v) in basisVectors.enumerated() {
      XCTAssertEqual(pb(v).0, Double(index + 1))
      XCTAssertEqual(pb(v).1, 2 * v)
    }
  }

  /// Tests the value and derivative of the functional *.
  func testScalarProductFunctional() {
    let s = Double(2)
    let v1 = makeVector(from: 1, stride: 1)
    XCTAssertEqual(s * v1, makeVector(from: 2, stride: 2))

    let (value, pb) = valueWithPullback(at: s, v1, in: *)
    XCTAssertEqual(value, makeVector(from: 2, stride: 2))
    for (index, v) in basisVectors.enumerated() {
      XCTAssertEqual(pb(v).0, Double(index + 1))
      XCTAssertEqual(pb(v).1, 2 * v)
    }
  }

  /// Tests the value and derivative of `dot`.
  func testDot() {
    let v1 = makeVector(from: 1, stride: 1)
    let v2 = makeVector(from: 2, stride: 1)
    let expectedDot = (0..<dimension).map { Double(($0 + 1) * ($0 + 2)) }.reduce(0, +)
    XCTAssertEqual(v1.dot(v2), expectedDot)

    let (value, g) = valueWithGradient(at: v1, v2) { $0.dot($1) }
    XCTAssertEqual(value, expectedDot)
    XCTAssertEqual(g.0, v2)
    XCTAssertEqual(g.1, v1)
  }

  /// Tests the value and derivative of the prefix unary -.
  func testNegate() {
    let v1 = makeVector(from: 1, stride: 1)
    XCTAssertEqual(-v1, makeVector(from: -1, stride: -1))

    let (value, pb) = valueWithPullback(at: v1) { -$0 }
    XCTAssertEqual(value, makeVector(from: -1, stride: -1))
    for v in basisVectors {
      XCTAssertEqual(pb(v), -v)
    }
  }

  /// Tests the value and derivative of `squaredNorm`.
  func testSquaredNorm() {
    let v1 = makeVector(from: 1, stride: 1)
    let expectedSquaredNorm = (0..<dimension).map { Double(($0 + 1) * ($0 + 1)) }.reduce(0, +)
    XCTAssertEqual(v1.squaredNorm, expectedSquaredNorm)

    let (value, g) = valueWithGradient(at: v1) { $0.squaredNorm }
    XCTAssertEqual(value, expectedSquaredNorm)
    XCTAssertEqual(g, 2 * v1)
  }

  /// Tests the value and derivative of `norm`.
  func testNorm() {
    let v1 = makeVector(from: 1, stride: 1)
    let expectedNorm = (0..<dimension).map { Double(($0 + 1) * ($0 + 1)) }.reduce(0, +).squareRoot()
    XCTAssertEqual(v1.norm, expectedNorm)

    let (value, g) = valueWithGradient(at: v1) { $0.norm }
    XCTAssertEqual(value, expectedNorm)
    XCTAssertEqual(g, (1 / v1.norm) * v1)
  }
}
