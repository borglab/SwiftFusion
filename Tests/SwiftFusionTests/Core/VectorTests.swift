import _Differentiation
import Foundation
import PenguinTesting
import TensorFlow
import XCTest

import SwiftFusion
import PenguinStructures
import PenguinTesting

extension Vector {
  /// XCTests `self`'s semantic conformance to `Vector`, expecting its scalars to match
  /// `expectedScalars`.
  ///
  /// - Parameter distinctScalars: scalars that are elementwise distinct from `expectedScalars`.
  ///   These are written to the scalars collection to test mutable collection semantics.
  /// - Parameter maxSupportedScalarCount: the maximum number of scalars  that instances of `Self`
  ///   can have.
  /// - Requires: `!distinctScalars.elementsEqual(self.scalars)`.
  /// - Complexity: O(N²), where N is `self.dimension`.
  func checkVectorSemantics<S1: Collection, S2: Collection>(
    expectingScalars expectedScalars: S1,
    writingScalars distinctScalars: S2,
    maxSupportedScalarCount: Int = Int.max
  ) where S1.Element == Double, S2.Element == Double {

    // MARK: - Check the `scalars` property semantics.

    self.scalars.checkCollectionSemantics(
      expecting: expectedScalars, maxSupportedCount: maxSupportedScalarCount)

    var mutableScalars = self.scalars
    mutableScalars.checkMutableCollectionSemantics(writing: distinctScalars)

    // Check that setting `scalars` actually changes it.
    var mutableSelf = self
    mutableSelf.scalars = mutableScalars
    XCTAssertTrue(mutableSelf.scalars.elementsEqual(mutableScalars))

    // MARK: - Check the `dimension` property semantics.

    XCTAssertEqual(self.dimension, expectedScalars.count)

    // MARK: - Check the mathematical vector operation (`+`, `+=`, `-`, `-=`, `*`, `*=`, `dot`,
    // `zeroValue`) semantics.

    /// Returns `self` with its scalars replaced with `start`, `start + stride`,
    /// `start + 2 * stride`, etc.
    func strideVector(from start: Double, by stride: Double) -> Self {
      var r = self
      for (i, e) in zip(
        r.scalars.indices,
        Swift.stride(from: start, to: start + Double(r.dimension) * stride, by: stride)
      ) {
        r.scalars[i] = e
      }
      return r
    }

    let v1 = strideVector(from: 1, by: 1)
    let v2 = strideVector(from: 10, by: 10)

    // Check that the additive identity (`zeroValue`) satisfies identities.
    zeroValue.checkPlus(zeroValue, equals: zeroValue)
    v1.checkPlus(zeroValue, equals: v1)
    zeroValue.checkPlus(v1, equals: v1)
    v1.checkPlus(-v1, equals: zeroValue)
    zeroValue.checkMinus(zeroValue, equals: zeroValue)
    v1.checkMinus(zeroValue, equals: v1)
    zeroValue.checkMinus(v1, equals: -v1)
    v1.checkMinus(v1, equals: zeroValue)
    v1.checkDot(zeroValue, equals: 0)
    zeroValue.checkDot(v1, equals: 0)

    // Check that scalar multiplication satisfies identities.
    v1.checkTimes(0, equals: zeroValue)
    v1.checkTimes(1, equals: v1)
    v1.checkTimes(-1, equals: -v1)
    // Check that addition gives the expected result.
    let expectedV1PlusV2 = strideVector(from: 11, by: 11)
    v1.checkPlus(v2, equals: expectedV1PlusV2)
    v2.checkPlus(v1, equals: expectedV1PlusV2)
    // Check that subtraction gives the expected result.
    let expectedV1MinusV2 = strideVector(from: -9, by: -9)
    v1.checkMinus(v2, equals: expectedV1MinusV2)
    v2.checkMinus(v1, equals: -expectedV1MinusV2)
    // Check that dot product gives the expected result.
    let expectedV1DotV2 =
        Double(10 * self.dimension * (self.dimension + 1) * (2 * self.dimension + 1) / 6)
    v1.checkDot(v2, equals: expectedV1DotV2)
    v2.checkDot(v1, equals: expectedV1DotV2)

    // Check that scalar multiplication gives the expected result.
    v1.checkTimes(7, equals: strideVector(from: 7, by: 7))

    // MARK: - Check unsafe buffer semantics.

    self.withUnsafeBufferPointer { b in
      XCTAssertTrue(b.elementsEqual(expectedScalars))
    }

    mutableSelf = self
    mutableSelf.withUnsafeMutableBufferPointer { b in
      for (i, j) in zip(b.indices, distinctScalars.indices) {
        b[i] = distinctScalars[j]
      }
    }
    XCTAssertTrue(mutableSelf.scalars.elementsEqual(distinctScalars))
  }

  /// XCTests the semantics of the mutating and nonmutating addition operations at `(self, other)`,
  /// and their derivatives.
  private func checkPlus(_ other: Self, equals expectedResult: Self) {
    // Do the checks in a closure so that we can check both the mutating and nonmutating versions.
    func check(_ f: @differentiable (Self, Self) -> Self) {
      XCTAssertEqual(f(self, other), expectedResult)

      let (result, pb) = valueWithPullback(at: self, other, in: f)
      XCTAssertEqual(result, expectedResult)
      for v in StandardBasis(shapedLikeZero: result.zeroValue) {
        XCTAssertEqual(pb(v).0, v)
        XCTAssertEqual(pb(v).1, v)
      }
    }

    check { $0 + $1 }
    check {
      var r = $0
      r += $1
      return r
    }
  }

  /// XCTests the semantics of the mutating and nonmutating subtraction operations at `(self, other)`,
  /// and their derivatives.
  private func checkMinus(_ other: Self, equals expectedResult: Self) {
    // Do the checks in a closure so that we can check both the mutating and nonmutating versions.
    func check(_ f: @differentiable (Self, Self) -> Self) {
      XCTAssertEqual(f(self, other), expectedResult)

      let (result, pb) = valueWithPullback(at: self, other, in: f)
      XCTAssertEqual(result, expectedResult)
      for v in StandardBasis(shapedLikeZero: result.zeroValue) {
        XCTAssertEqual(pb(v).0, v)
        XCTAssertEqual(pb(v).1, -v)
      }
    }

    check { $0 - $1 }
    check {
      var r = $0
      r -= $1
      return r
    }
  }

  /// XCTests the semantics of the mutating and nonmutating scalar multiplication operations at
  /// `(self, scaleFactor)`, and their derivatives.
  private func checkTimes(_ scaleFactor: Double, equals expectedResult: Self) {
    // Do the checks in a closure so that we can check both the mutating and nonmutating versions.
    func check(_ f: @differentiable (Double, Self) -> Self) {
      XCTAssertEqual(f(scaleFactor, self), expectedResult)

      let (result, pb) = valueWithPullback(at: scaleFactor, self, in: f)
      XCTAssertEqual(result, expectedResult)
      for v in StandardBasis(shapedLikeZero: result.zeroValue) {
        XCTAssertEqual(pb(v).0, v.dot(self))
        XCTAssertEqual(pb(v).1, scaleFactor * v)
      }
    }

    check { $0 * $1 }
    check {
      var r = $1
      r *= $0
      return r
    }
  }

  /// XCTests the semantics of the inner product operation at `(self, other)`, and its derivative.
  private func checkDot(_ other: Self, equals expectedResult: Double) {
    XCTAssertEqual(self.dot(other), expectedResult)

    let (result, pb) = valueWithPullback(at: self, other) { $0.dot($1) }
    XCTAssertEqual(result, expectedResult)
    XCTAssertEqual(pb(1).0, other)
    XCTAssertEqual(pb(1).1, self)
  }
}

class VectorExtensionTests: XCTestCase {
  /// Tests converting from one type to another type with the same number of elements.
  func testConversion() {
    let v = Vector9(0, 1, 2, 3, 4, 5, 6, 7, 8)
    let m = Matrix3(0, 1, 2, 3, 4, 5, 6, 7, 8)
    XCTAssertEqual(Vector9(m), v)

    let (value, pb) = valueWithPullback(at: m) { Vector9($0) }
    XCTAssertEqual(value, v)
    for (bV, bM) in zip(Vector9.standardBasis, Matrix3.standardBasis) {
      XCTAssertEqual(pb(bV), bM)
    }
  }

  /// Tests concatenating two vectors.
  func testConcatenate() {
    let v1 = Vector2(0, 1)
    let v2 = Vector3(2, 3, 4)
    let expected = Vector5(0, 1, 2, 3, 4)
    XCTAssertEqual(Vector5(concatenating: v1, v2), expected)

    let (value, pb) = valueWithPullback(at: v1, v2) { Vector5(concatenating: $0, $1) }
    XCTAssertEqual(value, expected)

    XCTAssertEqual(pb(Vector5(1, 0, 0, 0, 0)).0, Vector2(1, 0))
    XCTAssertEqual(pb(Vector5(0, 1, 0, 0, 0)).0, Vector2(0, 1))
    XCTAssertEqual(pb(Vector5(0, 0, 1, 0, 0)).0, Vector2(0, 0))
    XCTAssertEqual(pb(Vector5(0, 0, 0, 1, 0)).0, Vector2(0, 0))
    XCTAssertEqual(pb(Vector5(0, 0, 0, 0, 1)).0, Vector2(0, 0))

    XCTAssertEqual(pb(Vector5(1, 0, 0, 0, 0)).1, Vector3(0, 0, 0))
    XCTAssertEqual(pb(Vector5(0, 1, 0, 0, 0)).1, Vector3(0, 0, 0))
    XCTAssertEqual(pb(Vector5(0, 0, 1, 0, 0)).1, Vector3(1, 0, 0))
    XCTAssertEqual(pb(Vector5(0, 0, 0, 1, 0)).1, Vector3(0, 1, 0))
    XCTAssertEqual(pb(Vector5(0, 0, 0, 0, 1)).1, Vector3(0, 0, 1))
  }

  func testConvertToTensor() {
    let v = Vector3(1, 2, 3)
    let expectedT = Tensor<Double>([1, 2, 3])
    XCTAssertEqual(v.flatTensor, expectedT)

    let (value, pb) = valueWithPullback(at: v) { $0.flatTensor }
    XCTAssertEqual(value, expectedT)
    XCTAssertEqual(pb(Tensor([1, 0, 0])), Vector3(1, 0, 0))
    XCTAssertEqual(pb(Tensor([0, 1, 0])), Vector3(0, 1, 0))
    XCTAssertEqual(pb(Tensor([0, 0, 1])), Vector3(0, 0, 1))
  }

  func testConvertFromTensor() {
    let t = Tensor<Double>([1, 2, 3])
    let expectedV = Vector3(1, 2, 3)
    XCTAssertEqual(Vector3(flatTensor: t), expectedV)

    let (value, pb) = valueWithPullback(at: t) { Vector3(flatTensor: $0) }
    XCTAssertEqual(value, expectedV)
    XCTAssertEqual(pb(Vector3(1, 0, 0)), Tensor([1, 0, 0]))
    XCTAssertEqual(pb(Vector3(0, 1, 0)), Tensor([0, 1, 0]))
    XCTAssertEqual(pb(Vector3(0, 0, 1)), Tensor([0, 0, 1]))
  }
}
