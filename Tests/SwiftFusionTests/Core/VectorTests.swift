// WARNING: This is a generated file. Do not edit it. Instead, edit the corresponding ".gyb" file.
// See "generate.sh" in the root of this repository for instructions how to regenerate files.

// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 1)
import Foundation
import TensorFlow
import XCTest

import SwiftFusion

// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 9)

class VectorTests: XCTestCase {
  static var allTests = [
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 13)
    ("testVector1Init", testVector1Init),
    ("testVector1Equality", testVector1Equality),
    ("testVector1Magnitude", testVector1Magnitude),
    ("testVector1Add", testVector1Add),
    ("testVector1Subtract", testVector1Subtract),
    ("testVector1ScalarMultiply", testVector1ScalarMultiply),
    ("testVector1Negate", testVector1Negate),
    ("testVector1TangentVector", testVector1TangentVector),
    ("testVector1Move", testVector1Move),
    ("testVector1TensorInit", testVector1TensorInit),
    ("testVector1TensorExtract", testVector1TensorExtract),
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 13)
    ("testVector2Init", testVector2Init),
    ("testVector2Equality", testVector2Equality),
    ("testVector2Magnitude", testVector2Magnitude),
    ("testVector2Add", testVector2Add),
    ("testVector2Subtract", testVector2Subtract),
    ("testVector2ScalarMultiply", testVector2ScalarMultiply),
    ("testVector2Negate", testVector2Negate),
    ("testVector2TangentVector", testVector2TangentVector),
    ("testVector2Move", testVector2Move),
    ("testVector2TensorInit", testVector2TensorInit),
    ("testVector2TensorExtract", testVector2TensorExtract),
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 13)
    ("testVector3Init", testVector3Init),
    ("testVector3Equality", testVector3Equality),
    ("testVector3Magnitude", testVector3Magnitude),
    ("testVector3Add", testVector3Add),
    ("testVector3Subtract", testVector3Subtract),
    ("testVector3ScalarMultiply", testVector3ScalarMultiply),
    ("testVector3Negate", testVector3Negate),
    ("testVector3TangentVector", testVector3TangentVector),
    ("testVector3Move", testVector3Move),
    ("testVector3TensorInit", testVector3TensorInit),
    ("testVector3TensorExtract", testVector3TensorExtract)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 25)
  ]

// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 31)

  /// Test that initializing a vector from coordinate values works.
  func testVector1Init() {
    let vector1 = Vector1(1)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 36)
    XCTAssertEqual(vector1.x, 1)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 38)
  }

  /// Test that vector magnitude works.
  func testVector1Magnitude() {
    let vector1 = Vector1(1)
    XCTAssertEqual(vector1.magnitude, 1.0, accuracy: 1e-6)
  }

  /// Test that vector `==` works.
  func testVector1Equality() {
    let vector1 = Vector1(1)
    let vector2 = Vector1(2)
    XCTAssertTrue(vector1 == vector1)
    XCTAssertFalse(vector1 == vector2)
  }

  /// Test that vector addition works.
  func testVector1Add() {
    let vector1 = Vector1(1)
    let vector2 = Vector1(2)
    let sum = vector1 + vector2
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 60)
    XCTAssertEqual(sum.x, 3)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 62)
  }

  /// Test that vector subtraction works.
  func testVector1Subtract() {
    let vector1 = Vector1(1)
    let vector2 = Vector1(2)
    let difference = vector1 - vector2
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 70)
    XCTAssertEqual(difference.x, -1)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 72)
  }

  /// Test that vector scalar multiplication works.
  func testVector1ScalarMultiply() {
    let vector1 = Vector1(1)
    let scaled = vector1.scaled(by: 2)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 79)
    XCTAssertEqual(scaled.x, 2)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 81)
  }

  /// Test that vector negation works.
  func testVector1Negate() {
    let vector1 = Vector1(1)
    let negated = -vector1
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 88)
    XCTAssertEqual(negated.x, -1)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 90)
  }

  /// Tests that `Vector1.TangentVector == Vector1`.
  func testVector1TangentVector() {
    let vector1 = Vector1(1)
    let _: Vector1.TangentVector = vector1
  }

  /// Tests that the move (exponential map) operation works on vectors.
  func testVector1Move() {
    let vector1 = Vector1(1)
    let vector2 = Vector1(2)
    var moved = vector1
    moved.move(along: vector2)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 105)
    XCTAssertEqual(moved.x, 3)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 107)
  }

  /// Tests that we can initialize a vector from a tensor.
  func testVector1TensorInit() {
    let vector1 = Vector1(1)
    let tensor1 = Tensor<Double>([1])
    XCTAssertEqual(Vector1(tensor1), vector1)
  }

  /// Tests that we can extract a tensor from a vector.
  func testVector1TensorExtract() {
    let vector1 = Vector1(1)
    let tensor1 = Tensor<Double>([1])
    XCTAssertEqual(vector1.tensor, tensor1)
  }

// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 31)

  /// Test that initializing a vector from coordinate values works.
  func testVector2Init() {
    let vector1 = Vector2(1, 2)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 36)
    XCTAssertEqual(vector1.x, 1)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 36)
    XCTAssertEqual(vector1.y, 2)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 38)
  }

  /// Test that vector magnitude works.
  func testVector2Magnitude() {
    let vector1 = Vector2(1, 2)
    XCTAssertEqual(vector1.magnitude, 2.23606797749979, accuracy: 1e-6)
  }

  /// Test that vector `==` works.
  func testVector2Equality() {
    let vector1 = Vector2(1, 2)
    let vector2 = Vector2(3, 4)
    XCTAssertTrue(vector1 == vector1)
    XCTAssertFalse(vector1 == vector2)
  }

  /// Test that vector addition works.
  func testVector2Add() {
    let vector1 = Vector2(1, 2)
    let vector2 = Vector2(3, 4)
    let sum = vector1 + vector2
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 60)
    XCTAssertEqual(sum.x, 4)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 60)
    XCTAssertEqual(sum.y, 6)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 62)
  }

  /// Test that vector subtraction works.
  func testVector2Subtract() {
    let vector1 = Vector2(1, 2)
    let vector2 = Vector2(3, 4)
    let difference = vector1 - vector2
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 70)
    XCTAssertEqual(difference.x, -2)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 70)
    XCTAssertEqual(difference.y, -2)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 72)
  }

  /// Test that vector scalar multiplication works.
  func testVector2ScalarMultiply() {
    let vector1 = Vector2(1, 2)
    let scaled = vector1.scaled(by: 2)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 79)
    XCTAssertEqual(scaled.x, 2)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 79)
    XCTAssertEqual(scaled.y, 4)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 81)
  }

  /// Test that vector negation works.
  func testVector2Negate() {
    let vector1 = Vector2(1, 2)
    let negated = -vector1
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 88)
    XCTAssertEqual(negated.x, -1)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 88)
    XCTAssertEqual(negated.y, -2)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 90)
  }

  /// Tests that `Vector2.TangentVector == Vector2`.
  func testVector2TangentVector() {
    let vector1 = Vector2(1, 2)
    let _: Vector2.TangentVector = vector1
  }

  /// Tests that the move (exponential map) operation works on vectors.
  func testVector2Move() {
    let vector1 = Vector2(1, 2)
    let vector2 = Vector2(3, 4)
    var moved = vector1
    moved.move(along: vector2)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 105)
    XCTAssertEqual(moved.x, 4)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 105)
    XCTAssertEqual(moved.y, 6)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 107)
  }

  /// Tests that we can initialize a vector from a tensor.
  func testVector2TensorInit() {
    let vector1 = Vector2(1, 2)
    let tensor1 = Tensor<Double>([1, 2])
    XCTAssertEqual(Vector2(tensor1), vector1)
  }

  /// Tests that we can extract a tensor from a vector.
  func testVector2TensorExtract() {
    let vector1 = Vector2(1, 2)
    let tensor1 = Tensor<Double>([1, 2])
    XCTAssertEqual(vector1.tensor, tensor1)
  }

// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 31)

  /// Test that initializing a vector from coordinate values works.
  func testVector3Init() {
    let vector1 = Vector3(1, 2, 3)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 36)
    XCTAssertEqual(vector1.x, 1)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 36)
    XCTAssertEqual(vector1.y, 2)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 36)
    XCTAssertEqual(vector1.z, 3)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 38)
  }

  /// Test that vector magnitude works.
  func testVector3Magnitude() {
    let vector1 = Vector3(1, 2, 3)
    XCTAssertEqual(vector1.magnitude, 3.7416573867739413, accuracy: 1e-6)
  }

  /// Test that vector `==` works.
  func testVector3Equality() {
    let vector1 = Vector3(1, 2, 3)
    let vector2 = Vector3(4, 5, 6)
    XCTAssertTrue(vector1 == vector1)
    XCTAssertFalse(vector1 == vector2)
  }

  /// Test that vector addition works.
  func testVector3Add() {
    let vector1 = Vector3(1, 2, 3)
    let vector2 = Vector3(4, 5, 6)
    let sum = vector1 + vector2
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 60)
    XCTAssertEqual(sum.x, 5)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 60)
    XCTAssertEqual(sum.y, 7)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 60)
    XCTAssertEqual(sum.z, 9)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 62)
  }

  /// Test that vector subtraction works.
  func testVector3Subtract() {
    let vector1 = Vector3(1, 2, 3)
    let vector2 = Vector3(4, 5, 6)
    let difference = vector1 - vector2
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 70)
    XCTAssertEqual(difference.x, -3)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 70)
    XCTAssertEqual(difference.y, -3)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 70)
    XCTAssertEqual(difference.z, -3)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 72)
  }

  /// Test that vector scalar multiplication works.
  func testVector3ScalarMultiply() {
    let vector1 = Vector3(1, 2, 3)
    let scaled = vector1.scaled(by: 2)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 79)
    XCTAssertEqual(scaled.x, 2)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 79)
    XCTAssertEqual(scaled.y, 4)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 79)
    XCTAssertEqual(scaled.z, 6)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 81)
  }

  /// Test that vector negation works.
  func testVector3Negate() {
    let vector1 = Vector3(1, 2, 3)
    let negated = -vector1
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 88)
    XCTAssertEqual(negated.x, -1)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 88)
    XCTAssertEqual(negated.y, -2)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 88)
    XCTAssertEqual(negated.z, -3)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 90)
  }

  /// Tests that `Vector3.TangentVector == Vector3`.
  func testVector3TangentVector() {
    let vector1 = Vector3(1, 2, 3)
    let _: Vector3.TangentVector = vector1
  }

  /// Tests that the move (exponential map) operation works on vectors.
  func testVector3Move() {
    let vector1 = Vector3(1, 2, 3)
    let vector2 = Vector3(4, 5, 6)
    var moved = vector1
    moved.move(along: vector2)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 105)
    XCTAssertEqual(moved.x, 5)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 105)
    XCTAssertEqual(moved.y, 7)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 105)
    XCTAssertEqual(moved.z, 9)
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 107)
  }

  /// Tests that we can initialize a vector from a tensor.
  func testVector3TensorInit() {
    let vector1 = Vector3(1, 2, 3)
    let tensor1 = Tensor<Double>([1, 2, 3])
    XCTAssertEqual(Vector3(tensor1), vector1)
  }

  /// Tests that we can extract a tensor from a vector.
  func testVector3TensorExtract() {
    let vector1 = Vector3(1, 2, 3)
    let tensor1 = Tensor<Double>([1, 2, 3])
    XCTAssertEqual(vector1.tensor, tensor1)
  }

// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Tests/SwiftFusionTests/Core/VectorTests.swift.gyb", line: 124)
}
