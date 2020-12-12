// Copyright 2020 The SwiftFusion Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import _Differentiation
import Foundation
import TensorFlow
import XCTest

import PenguinStructures
@testable import SwiftFusion

class ValuesStorageTests: XCTestCase {
  func testDifferentiableTangentVectorType() {
    let values = AnyDifferentiableArrayBuffer(ArrayBuffer([Pose2(0, 0, 0), Pose2(1, 0, 0)]))
    XCTAssertEqual(
      ObjectIdentifier(values.tangentVectorType), ObjectIdentifier(Pose2.TangentVector.self))
  }

  func testDifferentiableTangentVectorZeros() {
    let values = AnyDifferentiableArrayBuffer(ArrayBuffer([Pose2(0, 0, 0), Pose2(1, 0, 0)]))
    assertElementsEqual(values.tangentVectorZeros, (0..<2).map { _ in Vector3.zero })
  }

  func testDifferentiableMove() {
    var values = AnyDifferentiableArrayBuffer(ArrayBuffer([Pose2(0, 0, 0), Pose2(1, 0, 0)]))
    let directions = AnyElementArrayBuffer(ArrayBuffer([Vector3(0, 0, 0), Vector3(0, 0, 1)]))
    values.move(along: directions)
    assertElementsEqual(values, [Pose2(0, 0, 0), Pose2(1, 1, 0)])
  }

  func testVectorTangentVectorZeros() {
    let v1 = AnyVectorArrayBuffer(ArrayBuffer([Vector3(1, 2, 3), Vector3(4, 5, 6)]))
    assertElementsEqual(v1.tangentVectorZeros, (0..<2).map { _ in Vector3(0, 0, 0) })
  }

  func testVectorMove() {
    var v1 = AnyVectorArrayBuffer(ArrayBuffer([Vector3(1, 2, 3), Vector3(4, 5, 6)]))
    let v2 = AnyElementArrayBuffer(ArrayBuffer([Vector3(1, 1, 1), Vector3(2, 2, 2)]))
    v1.move(along: v2)
    assertElementsEqual(v1, [Vector3(2, 3, 4), Vector3(6, 7, 8)])
  }

  func testVectorAdd() {
    var v1 = AnyVectorArrayBuffer(ArrayBuffer([Vector3(1, 2, 3), Vector3(4, 5, 6)]))
    let v2 = AnyElementArrayBuffer(ArrayBuffer([Vector3(1, 1, 1), Vector3(2, 2, 2)]))
    v1.add(v2)
    assertElementsEqual(v1, [Vector3(2, 3, 4), Vector3(6, 7, 8)])
  }

  func testVectorScale() {
    var v1 = AnyVectorArrayBuffer(ArrayBuffer([Vector3(1, 2, 3), Vector3(4, 5, 6)]))
    v1.scale(by: 10)
    assertElementsEqual(v1, [Vector3(10, 20, 30), Vector3(40, 50, 60)])
  }

  func testVectorDot() {
    let v1 = AnyVectorArrayBuffer(ArrayBuffer([Vector3(1, 2, 3), Vector3(4, 5, 6)]))
    let v2 = AnyVectorArrayBuffer(ArrayBuffer([Vector3(1, 1, 1), Vector3(2, 2, 2)]))
    XCTAssertEqual(v1.dot(v2), 36)
  }

  func assertElementsEqual<Dispatch, Elements>(
    _ actual: AnyArrayBuffer<Dispatch>,
    _ expected: Elements,
    file: StaticString = #filePath,
    line: UInt = #line
  ) where Elements: Collection, Elements.Element: Equatable {
    guard let typedActual = ArrayBuffer<Elements.Element>(actual) else {
      XCTFail(
        """
        Expected element type `\(Elements.Element.self)` but type-erased buffer has incompatible
        type `\(type(of: actual.storage!))`
        """, file: (file), line: line)
      return
    }
    XCTAssertEqual(Array(typedActual), Array(expected), file: (file), line: line)
  }
}
