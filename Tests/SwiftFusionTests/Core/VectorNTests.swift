// WARNING: This is a generated file. Do not edit it. Instead, edit the corresponding ".gyb" file.
// See "generate.sh" in the root of this repository for instructions how to regenerate files.

// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 1)
import Foundation
import TensorFlow
import XCTest

import SwiftFusion

// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 9)

class ConcreteEuclideanVectorNTests: XCTestCase {
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 19)

  /// Test that initializing a vector from coordinate values works.
  func testVector1Init() {
    let vector1 = Vector1(1)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.x, 1)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 26)
  }

  /// Test that the vector has the correct dimension.
  func testVector1Dimension() {
    XCTAssertEqual(Vector1.dimension, 1)
  }

  /// Test that vector `==` works.
  func testVector1Equality() {
    let vector1 = Vector1(1)
    let vector2 = Vector1(2)
    XCTAssertTrue(vector1 == vector1)
    XCTAssertFalse(vector1 == vector2)
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
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.x, 3)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 56)
  }

  /// Tests that conversion to `Vector` works.
  func testVector1ConvertToVector() {
    XCTAssertEqual(
      Vector1(1).vector,
      Vector([1])
    )
  }

  /// Tests that conversion from `Vector` works.
  func testVector1ConvertFromVector() {
    XCTAssertEqual(
      Vector1(Vector([1])),
      Vector1(1)
    )
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

// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 19)

  /// Test that initializing a vector from coordinate values works.
  func testVector2Init() {
    let vector1 = Vector2(1, 2)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.x, 1)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.y, 2)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 26)
  }

  /// Test that the vector has the correct dimension.
  func testVector2Dimension() {
    XCTAssertEqual(Vector2.dimension, 2)
  }

  /// Test that vector `==` works.
  func testVector2Equality() {
    let vector1 = Vector2(1, 2)
    let vector2 = Vector2(3, 4)
    XCTAssertTrue(vector1 == vector1)
    XCTAssertFalse(vector1 == vector2)
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
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.x, 4)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.y, 6)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 56)
  }

  /// Tests that conversion to `Vector` works.
  func testVector2ConvertToVector() {
    XCTAssertEqual(
      Vector2(1, 2).vector,
      Vector([1, 2])
    )
  }

  /// Tests that conversion from `Vector` works.
  func testVector2ConvertFromVector() {
    XCTAssertEqual(
      Vector2(Vector([1, 2])),
      Vector2(1, 2)
    )
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

// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 19)

  /// Test that initializing a vector from coordinate values works.
  func testVector3Init() {
    let vector1 = Vector3(1, 2, 3)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.x, 1)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.y, 2)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.z, 3)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 26)
  }

  /// Test that the vector has the correct dimension.
  func testVector3Dimension() {
    XCTAssertEqual(Vector3.dimension, 3)
  }

  /// Test that vector `==` works.
  func testVector3Equality() {
    let vector1 = Vector3(1, 2, 3)
    let vector2 = Vector3(4, 5, 6)
    XCTAssertTrue(vector1 == vector1)
    XCTAssertFalse(vector1 == vector2)
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
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.x, 5)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.y, 7)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.z, 9)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 56)
  }

  /// Tests that conversion to `Vector` works.
  func testVector3ConvertToVector() {
    XCTAssertEqual(
      Vector3(1, 2, 3).vector,
      Vector([1, 2, 3])
    )
  }

  /// Tests that conversion from `Vector` works.
  func testVector3ConvertFromVector() {
    XCTAssertEqual(
      Vector3(Vector([1, 2, 3])),
      Vector3(1, 2, 3)
    )
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

// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 19)

  /// Test that initializing a vector from coordinate values works.
  func testVector4Init() {
    let vector1 = Vector4(1, 2, 3, 4)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s0, 1)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s1, 2)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s2, 3)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s3, 4)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 26)
  }

  /// Test that the vector has the correct dimension.
  func testVector4Dimension() {
    XCTAssertEqual(Vector4.dimension, 4)
  }

  /// Test that vector `==` works.
  func testVector4Equality() {
    let vector1 = Vector4(1, 2, 3, 4)
    let vector2 = Vector4(5, 6, 7, 8)
    XCTAssertTrue(vector1 == vector1)
    XCTAssertFalse(vector1 == vector2)
  }

  /// Tests that `Vector4.TangentVector == Vector4`.
  func testVector4TangentVector() {
    let vector1 = Vector4(1, 2, 3, 4)
    let _: Vector4.TangentVector = vector1
  }

  /// Tests that the move (exponential map) operation works on vectors.
  func testVector4Move() {
    let vector1 = Vector4(1, 2, 3, 4)
    let vector2 = Vector4(5, 6, 7, 8)
    var moved = vector1
    moved.move(along: vector2)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s0, 6)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s1, 8)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s2, 10)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s3, 12)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 56)
  }

  /// Tests that conversion to `Vector` works.
  func testVector4ConvertToVector() {
    XCTAssertEqual(
      Vector4(1, 2, 3, 4).vector,
      Vector([1, 2, 3, 4])
    )
  }

  /// Tests that conversion from `Vector` works.
  func testVector4ConvertFromVector() {
    XCTAssertEqual(
      Vector4(Vector([1, 2, 3, 4])),
      Vector4(1, 2, 3, 4)
    )
  }

  /// Tests that we can initialize a vector from a tensor.
  func testVector4TensorInit() {
    let vector1 = Vector4(1, 2, 3, 4)
    let tensor1 = Tensor<Double>([1, 2, 3, 4])
    XCTAssertEqual(Vector4(tensor1), vector1)
  }

  /// Tests that we can extract a tensor from a vector.
  func testVector4TensorExtract() {
    let vector1 = Vector4(1, 2, 3, 4)
    let tensor1 = Tensor<Double>([1, 2, 3, 4])
    XCTAssertEqual(vector1.tensor, tensor1)
  }

// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 19)

  /// Test that initializing a vector from coordinate values works.
  func testVector5Init() {
    let vector1 = Vector5(1, 2, 3, 4, 5)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s0, 1)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s1, 2)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s2, 3)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s3, 4)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s4, 5)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 26)
  }

  /// Test that the vector has the correct dimension.
  func testVector5Dimension() {
    XCTAssertEqual(Vector5.dimension, 5)
  }

  /// Test that vector `==` works.
  func testVector5Equality() {
    let vector1 = Vector5(1, 2, 3, 4, 5)
    let vector2 = Vector5(6, 7, 8, 9, 10)
    XCTAssertTrue(vector1 == vector1)
    XCTAssertFalse(vector1 == vector2)
  }

  /// Tests that `Vector5.TangentVector == Vector5`.
  func testVector5TangentVector() {
    let vector1 = Vector5(1, 2, 3, 4, 5)
    let _: Vector5.TangentVector = vector1
  }

  /// Tests that the move (exponential map) operation works on vectors.
  func testVector5Move() {
    let vector1 = Vector5(1, 2, 3, 4, 5)
    let vector2 = Vector5(6, 7, 8, 9, 10)
    var moved = vector1
    moved.move(along: vector2)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s0, 7)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s1, 9)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s2, 11)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s3, 13)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s4, 15)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 56)
  }

  /// Tests that conversion to `Vector` works.
  func testVector5ConvertToVector() {
    XCTAssertEqual(
      Vector5(1, 2, 3, 4, 5).vector,
      Vector([1, 2, 3, 4, 5])
    )
  }

  /// Tests that conversion from `Vector` works.
  func testVector5ConvertFromVector() {
    XCTAssertEqual(
      Vector5(Vector([1, 2, 3, 4, 5])),
      Vector5(1, 2, 3, 4, 5)
    )
  }

  /// Tests that we can initialize a vector from a tensor.
  func testVector5TensorInit() {
    let vector1 = Vector5(1, 2, 3, 4, 5)
    let tensor1 = Tensor<Double>([1, 2, 3, 4, 5])
    XCTAssertEqual(Vector5(tensor1), vector1)
  }

  /// Tests that we can extract a tensor from a vector.
  func testVector5TensorExtract() {
    let vector1 = Vector5(1, 2, 3, 4, 5)
    let tensor1 = Tensor<Double>([1, 2, 3, 4, 5])
    XCTAssertEqual(vector1.tensor, tensor1)
  }

// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 19)

  /// Test that initializing a vector from coordinate values works.
  func testVector6Init() {
    let vector1 = Vector6(1, 2, 3, 4, 5, 6)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s0, 1)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s1, 2)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s2, 3)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s3, 4)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s4, 5)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s5, 6)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 26)
  }

  /// Test that the vector has the correct dimension.
  func testVector6Dimension() {
    XCTAssertEqual(Vector6.dimension, 6)
  }

  /// Test that vector `==` works.
  func testVector6Equality() {
    let vector1 = Vector6(1, 2, 3, 4, 5, 6)
    let vector2 = Vector6(7, 8, 9, 10, 11, 12)
    XCTAssertTrue(vector1 == vector1)
    XCTAssertFalse(vector1 == vector2)
  }

  /// Tests that `Vector6.TangentVector == Vector6`.
  func testVector6TangentVector() {
    let vector1 = Vector6(1, 2, 3, 4, 5, 6)
    let _: Vector6.TangentVector = vector1
  }

  /// Tests that the move (exponential map) operation works on vectors.
  func testVector6Move() {
    let vector1 = Vector6(1, 2, 3, 4, 5, 6)
    let vector2 = Vector6(7, 8, 9, 10, 11, 12)
    var moved = vector1
    moved.move(along: vector2)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s0, 8)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s1, 10)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s2, 12)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s3, 14)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s4, 16)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s5, 18)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 56)
  }

  /// Tests that conversion to `Vector` works.
  func testVector6ConvertToVector() {
    XCTAssertEqual(
      Vector6(1, 2, 3, 4, 5, 6).vector,
      Vector([1, 2, 3, 4, 5, 6])
    )
  }

  /// Tests that conversion from `Vector` works.
  func testVector6ConvertFromVector() {
    XCTAssertEqual(
      Vector6(Vector([1, 2, 3, 4, 5, 6])),
      Vector6(1, 2, 3, 4, 5, 6)
    )
  }

  /// Tests that we can initialize a vector from a tensor.
  func testVector6TensorInit() {
    let vector1 = Vector6(1, 2, 3, 4, 5, 6)
    let tensor1 = Tensor<Double>([1, 2, 3, 4, 5, 6])
    XCTAssertEqual(Vector6(tensor1), vector1)
  }

  /// Tests that we can extract a tensor from a vector.
  func testVector6TensorExtract() {
    let vector1 = Vector6(1, 2, 3, 4, 5, 6)
    let tensor1 = Tensor<Double>([1, 2, 3, 4, 5, 6])
    XCTAssertEqual(vector1.tensor, tensor1)
  }

// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 19)

  /// Test that initializing a vector from coordinate values works.
  func testVector7Init() {
    let vector1 = Vector7(1, 2, 3, 4, 5, 6, 7)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s0, 1)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s1, 2)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s2, 3)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s3, 4)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s4, 5)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s5, 6)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s6, 7)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 26)
  }

  /// Test that the vector has the correct dimension.
  func testVector7Dimension() {
    XCTAssertEqual(Vector7.dimension, 7)
  }

  /// Test that vector `==` works.
  func testVector7Equality() {
    let vector1 = Vector7(1, 2, 3, 4, 5, 6, 7)
    let vector2 = Vector7(8, 9, 10, 11, 12, 13, 14)
    XCTAssertTrue(vector1 == vector1)
    XCTAssertFalse(vector1 == vector2)
  }

  /// Tests that `Vector7.TangentVector == Vector7`.
  func testVector7TangentVector() {
    let vector1 = Vector7(1, 2, 3, 4, 5, 6, 7)
    let _: Vector7.TangentVector = vector1
  }

  /// Tests that the move (exponential map) operation works on vectors.
  func testVector7Move() {
    let vector1 = Vector7(1, 2, 3, 4, 5, 6, 7)
    let vector2 = Vector7(8, 9, 10, 11, 12, 13, 14)
    var moved = vector1
    moved.move(along: vector2)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s0, 9)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s1, 11)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s2, 13)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s3, 15)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s4, 17)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s5, 19)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s6, 21)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 56)
  }

  /// Tests that conversion to `Vector` works.
  func testVector7ConvertToVector() {
    XCTAssertEqual(
      Vector7(1, 2, 3, 4, 5, 6, 7).vector,
      Vector([1, 2, 3, 4, 5, 6, 7])
    )
  }

  /// Tests that conversion from `Vector` works.
  func testVector7ConvertFromVector() {
    XCTAssertEqual(
      Vector7(Vector([1, 2, 3, 4, 5, 6, 7])),
      Vector7(1, 2, 3, 4, 5, 6, 7)
    )
  }

  /// Tests that we can initialize a vector from a tensor.
  func testVector7TensorInit() {
    let vector1 = Vector7(1, 2, 3, 4, 5, 6, 7)
    let tensor1 = Tensor<Double>([1, 2, 3, 4, 5, 6, 7])
    XCTAssertEqual(Vector7(tensor1), vector1)
  }

  /// Tests that we can extract a tensor from a vector.
  func testVector7TensorExtract() {
    let vector1 = Vector7(1, 2, 3, 4, 5, 6, 7)
    let tensor1 = Tensor<Double>([1, 2, 3, 4, 5, 6, 7])
    XCTAssertEqual(vector1.tensor, tensor1)
  }

// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 19)

  /// Test that initializing a vector from coordinate values works.
  func testVector8Init() {
    let vector1 = Vector8(1, 2, 3, 4, 5, 6, 7, 8)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s0, 1)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s1, 2)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s2, 3)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s3, 4)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s4, 5)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s5, 6)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s6, 7)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s7, 8)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 26)
  }

  /// Test that the vector has the correct dimension.
  func testVector8Dimension() {
    XCTAssertEqual(Vector8.dimension, 8)
  }

  /// Test that vector `==` works.
  func testVector8Equality() {
    let vector1 = Vector8(1, 2, 3, 4, 5, 6, 7, 8)
    let vector2 = Vector8(9, 10, 11, 12, 13, 14, 15, 16)
    XCTAssertTrue(vector1 == vector1)
    XCTAssertFalse(vector1 == vector2)
  }

  /// Tests that `Vector8.TangentVector == Vector8`.
  func testVector8TangentVector() {
    let vector1 = Vector8(1, 2, 3, 4, 5, 6, 7, 8)
    let _: Vector8.TangentVector = vector1
  }

  /// Tests that the move (exponential map) operation works on vectors.
  func testVector8Move() {
    let vector1 = Vector8(1, 2, 3, 4, 5, 6, 7, 8)
    let vector2 = Vector8(9, 10, 11, 12, 13, 14, 15, 16)
    var moved = vector1
    moved.move(along: vector2)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s0, 10)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s1, 12)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s2, 14)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s3, 16)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s4, 18)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s5, 20)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s6, 22)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s7, 24)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 56)
  }

  /// Tests that conversion to `Vector` works.
  func testVector8ConvertToVector() {
    XCTAssertEqual(
      Vector8(1, 2, 3, 4, 5, 6, 7, 8).vector,
      Vector([1, 2, 3, 4, 5, 6, 7, 8])
    )
  }

  /// Tests that conversion from `Vector` works.
  func testVector8ConvertFromVector() {
    XCTAssertEqual(
      Vector8(Vector([1, 2, 3, 4, 5, 6, 7, 8])),
      Vector8(1, 2, 3, 4, 5, 6, 7, 8)
    )
  }

  /// Tests that we can initialize a vector from a tensor.
  func testVector8TensorInit() {
    let vector1 = Vector8(1, 2, 3, 4, 5, 6, 7, 8)
    let tensor1 = Tensor<Double>([1, 2, 3, 4, 5, 6, 7, 8])
    XCTAssertEqual(Vector8(tensor1), vector1)
  }

  /// Tests that we can extract a tensor from a vector.
  func testVector8TensorExtract() {
    let vector1 = Vector8(1, 2, 3, 4, 5, 6, 7, 8)
    let tensor1 = Tensor<Double>([1, 2, 3, 4, 5, 6, 7, 8])
    XCTAssertEqual(vector1.tensor, tensor1)
  }

// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 19)

  /// Test that initializing a vector from coordinate values works.
  func testVector9Init() {
    let vector1 = Vector9(1, 2, 3, 4, 5, 6, 7, 8, 9)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s0, 1)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s1, 2)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s2, 3)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s3, 4)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s4, 5)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s5, 6)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s6, 7)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s7, 8)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s8, 9)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 26)
  }

  /// Test that the vector has the correct dimension.
  func testVector9Dimension() {
    XCTAssertEqual(Vector9.dimension, 9)
  }

  /// Test that vector `==` works.
  func testVector9Equality() {
    let vector1 = Vector9(1, 2, 3, 4, 5, 6, 7, 8, 9)
    let vector2 = Vector9(10, 11, 12, 13, 14, 15, 16, 17, 18)
    XCTAssertTrue(vector1 == vector1)
    XCTAssertFalse(vector1 == vector2)
  }

  /// Tests that `Vector9.TangentVector == Vector9`.
  func testVector9TangentVector() {
    let vector1 = Vector9(1, 2, 3, 4, 5, 6, 7, 8, 9)
    let _: Vector9.TangentVector = vector1
  }

  /// Tests that the move (exponential map) operation works on vectors.
  func testVector9Move() {
    let vector1 = Vector9(1, 2, 3, 4, 5, 6, 7, 8, 9)
    let vector2 = Vector9(10, 11, 12, 13, 14, 15, 16, 17, 18)
    var moved = vector1
    moved.move(along: vector2)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s0, 11)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s1, 13)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s2, 15)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s3, 17)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s4, 19)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s5, 21)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s6, 23)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s7, 25)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 54)
    XCTAssertEqual(moved.s8, 27)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 56)
  }

  /// Tests that conversion to `Vector` works.
  func testVector9ConvertToVector() {
    XCTAssertEqual(
      Vector9(1, 2, 3, 4, 5, 6, 7, 8, 9).vector,
      Vector([1, 2, 3, 4, 5, 6, 7, 8, 9])
    )
  }

  /// Tests that conversion from `Vector` works.
  func testVector9ConvertFromVector() {
    XCTAssertEqual(
      Vector9(Vector([1, 2, 3, 4, 5, 6, 7, 8, 9])),
      Vector9(1, 2, 3, 4, 5, 6, 7, 8, 9)
    )
  }

  /// Tests that we can initialize a vector from a tensor.
  func testVector9TensorInit() {
    let vector1 = Vector9(1, 2, 3, 4, 5, 6, 7, 8, 9)
    let tensor1 = Tensor<Double>([1, 2, 3, 4, 5, 6, 7, 8, 9])
    XCTAssertEqual(Vector9(tensor1), vector1)
  }

  /// Tests that we can extract a tensor from a vector.
  func testVector9TensorExtract() {
    let vector1 = Vector9(1, 2, 3, 4, 5, 6, 7, 8, 9)
    let tensor1 = Tensor<Double>([1, 2, 3, 4, 5, 6, 7, 8, 9])
    XCTAssertEqual(vector1.tensor, tensor1)
  }

// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 89)
}

// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 92)
/// Tests the `EuclideanVectorN` requirements.
class Vector1EuclideanVectorNTests: XCTestCase, EuclideanVectorTests {
  typealias Testee = Vector1
  static var dimension: Int { return 1 }
  func testAll() {
    runAllEuclideanVectorNTests()
  }
}
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 92)
/// Tests the `EuclideanVectorN` requirements.
class Vector2EuclideanVectorNTests: XCTestCase, EuclideanVectorTests {
  typealias Testee = Vector2
  static var dimension: Int { return 2 }
  func testAll() {
    runAllEuclideanVectorNTests()
  }
}
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 92)
/// Tests the `EuclideanVectorN` requirements.
class Vector3EuclideanVectorNTests: XCTestCase, EuclideanVectorTests {
  typealias Testee = Vector3
  static var dimension: Int { return 3 }
  func testAll() {
    runAllEuclideanVectorNTests()
  }
}
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 92)
/// Tests the `EuclideanVectorN` requirements.
class Vector4EuclideanVectorNTests: XCTestCase, EuclideanVectorTests {
  typealias Testee = Vector4
  static var dimension: Int { return 4 }
  func testAll() {
    runAllEuclideanVectorNTests()
  }
}
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 92)
/// Tests the `EuclideanVectorN` requirements.
class Vector5EuclideanVectorNTests: XCTestCase, EuclideanVectorTests {
  typealias Testee = Vector5
  static var dimension: Int { return 5 }
  func testAll() {
    runAllEuclideanVectorNTests()
  }
}
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 92)
/// Tests the `EuclideanVectorN` requirements.
class Vector6EuclideanVectorNTests: XCTestCase, EuclideanVectorTests {
  typealias Testee = Vector6
  static var dimension: Int { return 6 }
  func testAll() {
    runAllEuclideanVectorNTests()
  }
}
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 92)
/// Tests the `EuclideanVectorN` requirements.
class Vector7EuclideanVectorNTests: XCTestCase, EuclideanVectorTests {
  typealias Testee = Vector7
  static var dimension: Int { return 7 }
  func testAll() {
    runAllEuclideanVectorNTests()
  }
}
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 92)
/// Tests the `EuclideanVectorN` requirements.
class Vector8EuclideanVectorNTests: XCTestCase, EuclideanVectorTests {
  typealias Testee = Vector8
  static var dimension: Int { return 8 }
  func testAll() {
    runAllEuclideanVectorNTests()
  }
}
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 92)
/// Tests the `EuclideanVectorN` requirements.
class Vector9EuclideanVectorNTests: XCTestCase, EuclideanVectorTests {
  typealias Testee = Vector9
  static var dimension: Int { return 9 }
  func testAll() {
    runAllEuclideanVectorNTests()
  }
}
