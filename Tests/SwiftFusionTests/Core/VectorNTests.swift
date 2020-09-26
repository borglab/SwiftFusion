// WARNING: This is a generated file. Do not edit it. Instead, edit the corresponding ".gyb" file.
// See "generate.sh" in the root of this repository for instructions how to regenerate files.

import Foundation
import TensorFlow
import XCTest

import SwiftFusion


class VectorNTests: XCTestCase {

  /// Test that initializing a vector from coordinate values works.
  func testVector1Init() {
    let vector1 = Vector1(1)
    XCTAssertEqual(vector1.x, 1)
  }

  func testVector1VectorConformance() {
    let s = (0..<1).lazy.map { Double($0) }
    let v = Vector1(0)
    v.checkVectorSemantics(
      expectingScalars: s,
      writingScalars: (1..<2).lazy.map { Double($0) },
      maxSupportedScalarCount: 1)
    v.scalars.checkRandomAccessCollectionSemantics(
      expecting: s,
      maxSupportedCount: 1)
  }

  /// Test that initializing a vector from coordinate values works.
  func testVector2Init() {
    let vector1 = Vector2(1, 2)
    XCTAssertEqual(vector1.x, 1)
    XCTAssertEqual(vector1.y, 2)
  }

  func testVector2VectorConformance() {
    let s = (0..<2).lazy.map { Double($0) }
    let v = Vector2(0, 1)
    v.checkVectorSemantics(
      expectingScalars: s,
      writingScalars: (2..<4).lazy.map { Double($0) },
      maxSupportedScalarCount: 2)
    v.scalars.checkRandomAccessCollectionSemantics(
      expecting: s,
      maxSupportedCount: 2)
  }

  /// Test that initializing a vector from coordinate values works.
  func testVector3Init() {
    let vector1 = Vector3(1, 2, 3)
    XCTAssertEqual(vector1.x, 1)
    XCTAssertEqual(vector1.y, 2)
    XCTAssertEqual(vector1.z, 3)
  }

  func testVector3VectorConformance() {
    let s = (0..<3).lazy.map { Double($0) }
    let v = Vector3(0, 1, 2)
    v.checkVectorSemantics(
      expectingScalars: s,
      writingScalars: (3..<6).lazy.map { Double($0) },
      maxSupportedScalarCount: 3)
    v.scalars.checkRandomAccessCollectionSemantics(
      expecting: s,
      maxSupportedCount: 3)
  }

  /// Test that initializing a vector from coordinate values works.
  func testVector4Init() {
    let vector1 = Vector4(1, 2, 3, 4)
    XCTAssertEqual(vector1.s0, 1)
    XCTAssertEqual(vector1.s1, 2)
    XCTAssertEqual(vector1.s2, 3)
    XCTAssertEqual(vector1.s3, 4)
  }

  func testVector4VectorConformance() {
    let s = (0..<4).lazy.map { Double($0) }
    let v = Vector4(0, 1, 2, 3)
    v.checkVectorSemantics(
      expectingScalars: s,
      writingScalars: (4..<8).lazy.map { Double($0) },
      maxSupportedScalarCount: 4)
    v.scalars.checkRandomAccessCollectionSemantics(
      expecting: s,
      maxSupportedCount: 4)
  }

  /// Test that initializing a vector from coordinate values works.
  func testVector5Init() {
    let vector1 = Vector5(1, 2, 3, 4, 5)
    XCTAssertEqual(vector1.s0, 1)
    XCTAssertEqual(vector1.s1, 2)
    XCTAssertEqual(vector1.s2, 3)
    XCTAssertEqual(vector1.s3, 4)
    XCTAssertEqual(vector1.s4, 5)
  }

  func testVector5VectorConformance() {
    let s = (0..<5).lazy.map { Double($0) }
    let v = Vector5(0, 1, 2, 3, 4)
    v.checkVectorSemantics(
      expectingScalars: s,
      writingScalars: (5..<10).lazy.map { Double($0) },
      maxSupportedScalarCount: 5)
    v.scalars.checkRandomAccessCollectionSemantics(
      expecting: s,
      maxSupportedCount: 5)
  }

  /// Test that initializing a vector from coordinate values works.
  func testVector6Init() {
    let vector1 = Vector6(1, 2, 3, 4, 5, 6)
    XCTAssertEqual(vector1.s0, 1)
    XCTAssertEqual(vector1.s1, 2)
    XCTAssertEqual(vector1.s2, 3)
    XCTAssertEqual(vector1.s3, 4)
    XCTAssertEqual(vector1.s4, 5)
    XCTAssertEqual(vector1.s5, 6)
  }

  func testVector6VectorConformance() {
    let s = (0..<6).lazy.map { Double($0) }
    let v = Vector6(0, 1, 2, 3, 4, 5)
    v.checkVectorSemantics(
      expectingScalars: s,
      writingScalars: (6..<12).lazy.map { Double($0) },
      maxSupportedScalarCount: 6)
    v.scalars.checkRandomAccessCollectionSemantics(
      expecting: s,
      maxSupportedCount: 6)
  }

  /// Test that initializing a vector from coordinate values works.
  func testVector7Init() {
    let vector1 = Vector7(1, 2, 3, 4, 5, 6, 7)
    XCTAssertEqual(vector1.s0, 1)
    XCTAssertEqual(vector1.s1, 2)
    XCTAssertEqual(vector1.s2, 3)
    XCTAssertEqual(vector1.s3, 4)
    XCTAssertEqual(vector1.s4, 5)
    XCTAssertEqual(vector1.s5, 6)
    XCTAssertEqual(vector1.s6, 7)
  }

  func testVector7VectorConformance() {
    let s = (0..<7).lazy.map { Double($0) }
    let v = Vector7(0, 1, 2, 3, 4, 5, 6)
    v.checkVectorSemantics(
      expectingScalars: s,
      writingScalars: (7..<14).lazy.map { Double($0) },
      maxSupportedScalarCount: 7)
    v.scalars.checkRandomAccessCollectionSemantics(
      expecting: s,
      maxSupportedCount: 7)
  }

  /// Test that initializing a vector from coordinate values works.
  func testVector8Init() {
    let vector1 = Vector8(1, 2, 3, 4, 5, 6, 7, 8)
    XCTAssertEqual(vector1.s0, 1)
    XCTAssertEqual(vector1.s1, 2)
    XCTAssertEqual(vector1.s2, 3)
    XCTAssertEqual(vector1.s3, 4)
    XCTAssertEqual(vector1.s4, 5)
    XCTAssertEqual(vector1.s5, 6)
    XCTAssertEqual(vector1.s6, 7)
    XCTAssertEqual(vector1.s7, 8)
  }

  func testVector8VectorConformance() {
    let s = (0..<8).lazy.map { Double($0) }
    let v = Vector8(0, 1, 2, 3, 4, 5, 6, 7)
    v.checkVectorSemantics(
      expectingScalars: s,
      writingScalars: (8..<16).lazy.map { Double($0) },
      maxSupportedScalarCount: 8)
    v.scalars.checkRandomAccessCollectionSemantics(
      expecting: s,
      maxSupportedCount: 8)
  }

  /// Test that initializing a vector from coordinate values works.
  func testVector9Init() {
    let vector1 = Vector9(1, 2, 3, 4, 5, 6, 7, 8, 9)
    XCTAssertEqual(vector1.s0, 1)
    XCTAssertEqual(vector1.s1, 2)
    XCTAssertEqual(vector1.s2, 3)
    XCTAssertEqual(vector1.s3, 4)
    XCTAssertEqual(vector1.s4, 5)
    XCTAssertEqual(vector1.s5, 6)
    XCTAssertEqual(vector1.s6, 7)
    XCTAssertEqual(vector1.s7, 8)
    XCTAssertEqual(vector1.s8, 9)
  }

  func testVector9VectorConformance() {
    let s = (0..<9).lazy.map { Double($0) }
    let v = Vector9(0, 1, 2, 3, 4, 5, 6, 7, 8)
    v.checkVectorSemantics(
      expectingScalars: s,
      writingScalars: (9..<18).lazy.map { Double($0) },
      maxSupportedScalarCount: 9)
    v.scalars.checkRandomAccessCollectionSemantics(
      expecting: s,
      maxSupportedCount: 9)
  }
}
