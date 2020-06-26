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

import Foundation
import TensorFlow
import XCTest

import PenguinStructures
import SwiftFusion

class VariableAssignmentsTests: XCTestCase {
  func testStoreAndSubscript() {
    var x = VariableAssignments()

    let intID = x.store(1)
    XCTAssertEqual(x[intID], 1)

    x[intID] = 2
    XCTAssertEqual(x[intID], 2)

    let doubleID = x.store(2.0)
    XCTAssertEqual(x[doubleID], 2.0)

    let vectorID = x.store(Vector2(3, 4))
    XCTAssertEqual(x[vectorID], Vector2(3, 4))

    print(x)
    // We desire something pretty like:
    //   [
    //     2,
    //     2.0,
    //     [3, 4]
    //   ]
    //
    // Or even better, named variables:
    //   [
    //     x1: 2,
    //     x2: 2.0,
    //     x3: [3, 4]
    //   ]
    //
    // But we get this instead:
    //   VariableAssignments(storage: [ObjectIdentifier(0x00007fcf9b7ea198): PenguinStructures.AnyArrayBuffer<SwiftFusion.AnyElementDispatch>(storage: Optional(PenguinStructures.ArrayStorage<Swift.Double>), dispatch: SwiftFusion.AnyElementDispatch), ObjectIdentifier(0x0000557603e35218): PenguinStructures.AnyArrayBuffer<SwiftFusion.AnyElementDispatch>(storage: Optional(PenguinStructures.ArrayStorage<SwiftFusion.Vector2>), dispatch: SwiftFusion.VectorArrayDispatch_<SwiftFusion.Vector2>), ObjectIdentifier(0x00007fcf9b7ea538): PenguinStructures.AnyArrayBuffer<SwiftFusion.AnyElementDispatch>(storage: Optional(PenguinStructures.ArrayStorage<Swift.Int>), dispatch: SwiftFusion.AnyElementDispatch)])
  }

  func testTangentVectorZeros() {
    var x = VariableAssignments()
    _ = x.store(1)
    _ = x.store(Vector1(2))
    _ = x.store(Vector2(3, 4))

    let t = x.tangentVectorZeros
    // TODO: Assert that there are only 2 elements when there is some API for count.
    XCTAssertEqual(t[TypedID<Vector1, Int>(0)], Vector1(0))
    XCTAssertEqual(t[TypedID<Vector2, Int>(0)], Vector2(0, 0))
  }

  func testMove() {
    var x = VariableAssignments()
    _ = x.store(1)
    let v1ID = x.store(Vector1(2))
    let v2ID = x.store(Vector2(3, 4))

    var t = VariableAssignments()
    _ = t.store(Vector1(10))
    _ = t.store(Vector2(20, 20))

    x.move(along: t)
    XCTAssertEqual(x[v1ID], Vector1(12))
    XCTAssertEqual(x[v2ID], Vector2(23, 24))
  }

  func testSquaredNorm() {
    var x = VariableAssignments()
    _ = x.store(Vector1(2))
    _ = x.store(Vector2(3, 4))
    XCTAssertEqual(x.squaredNorm, 29)
  }

  func testScalarMultiply() {
    var x = VariableAssignments()
    let v1ID = x.store(Vector1(2))
    let v2ID = x.store(Vector2(3, 4))

    let r = 10 * x
    XCTAssertEqual(r[v1ID], Vector1(20))
    XCTAssertEqual(r[v2ID], Vector2(30, 40))
  }

  func testPlus() {
    var x = VariableAssignments()
    let v1ID = x.store(Vector1(2))
    let v2ID = x.store(Vector2(3, 4))

    var t = VariableAssignments()
    _ = t.store(Vector1(10))
    _ = t.store(Vector2(20, 20))

    let r = x + t
    XCTAssertEqual(r[v1ID], Vector1(12))
    XCTAssertEqual(r[v2ID], Vector2(23, 24))
  }
}
