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
@testable import SwiftFusion

fileprivate typealias VectorArray<Element: EuclideanVectorN> =
  ArrayBuffer<VectorArrayStorage<Element>>
fileprivate typealias DifferentiableArray<Element: Differentiable>
  = ArrayBuffer<DifferentiableArrayStorage<Element>> where Element.TangentVector: EuclideanVectorN

class ValuesStorageTests: XCTestCase {
  
  func testDifferentiableMove() {
    var values = DifferentiableArray((0..<5).map { _ in Pose2(0, 0, 0) })
    let directions = VectorArray((0..<5).map { _ in Vector3(1, 0, 0) })
    values.move(along: directions)
    for i in 0..<5 {
      XCTAssertEqual(values[i], Pose2(0, 0, 1))
    }
  }
  
  func testVectorMove() {
    var values = VectorArray((0..<5).map { _ in Vector3(1, 2, 3) })
    let directions = VectorArray((0..<5).map { _ in Vector3(10, 20, 30) })
    values.move(along: directions)
    for i in 0..<5 {
      XCTAssertEqual(values[i], Vector3(11, 22, 33))
    }
  }

  func testVectorAdd() {
    var a = VectorArray((0..<5).map { _ in Vector3(1, 2, 3) })
    let b = VectorArray((0..<5).map { _ in Vector3(10, 20, 30) })
    a.add(b)
    for i in 0..<5 {
      XCTAssertEqual(a[i], Vector3(11, 22, 33))
    }
  }

  func testVectorScale() {
    var a = VectorArray((0..<5).map { _ in Vector3(1, 2, 3) })
    a.scale(by: 10)
    for i in 0..<5 {
      XCTAssertEqual(a[i], Vector3(10, 20, 30))
    }
  }

  func testVectorDot() {
    let a = VectorArray((0..<5).map { _ in Vector3(1, 2, 3) })
    let b = VectorArray((0..<5).map { _ in Vector3(10, 20, 30) })
    XCTAssertEqual(a.dot(b), 5 * (Double(1 * 10 + 2 * 20 + 3 * 30)))
  }
}
