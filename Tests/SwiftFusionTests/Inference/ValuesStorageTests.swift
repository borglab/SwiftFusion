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

/// Test helpers.
extension ArrayStorageImplementation {
  /// Creates an array with `count` copies of `element`.
  static func create(repeating element: Element, count: Int) -> Self {
    let array = Self.create(minimumCapacity: count)
    for _ in 0..<count { _ = array.append(element) }
    return array
  }
  
  /// Returns the element at `index`.
  subscript(index: Int) -> Element {
    return withUnsafeMutableBufferPointer { buffer in
      return buffer[index]
    }
  }
}

class ValuesStorageTests: XCTestCase {
  
  func testDifferentiableMove() {
    let values = DifferentiableArrayStorage.create(repeating: Pose2(0, 0, 0), count: 5)
    let directions = VectorArrayStorage.create(repeating: Vector3(1, 0, 0), count: 5)
    directions.withUnsafeMutableRawBufferPointer { directionBuffer in
      values.move(along: UnsafeRawBufferPointer(directionBuffer))
    }
    for i in 0..<5 {
      XCTAssertEqual(values[i], Pose2(0, 0, 1))
    }
  }
  
  func testVectorMove() {
    let values = VectorArrayStorage.create(repeating: Vector3(1, 2, 3), count: 5)
    let directions = VectorArrayStorage.create(repeating: Vector3(10, 20, 30), count: 5)
    directions.withUnsafeMutableRawBufferPointer { directionBuffer in
      values.move(along: UnsafeRawBufferPointer(directionBuffer))
    }
    for i in 0..<5 {
      XCTAssertEqual(values[i], Vector3(11, 22, 33))
    }
  }
  
  func testVectorAdd() {
    let a = VectorArrayStorage.create(repeating: Vector3(1, 2, 3), count: 5)
    let b = VectorArrayStorage.create(repeating: Vector3(10, 20, 30), count: 5)
    b.withUnsafeMutableBufferPointer { bBuffer in
      a.add(UnsafeRawBufferPointer(bBuffer))
    }
    for i in 0..<5 {
      XCTAssertEqual(a[i], Vector3(11, 22, 33))
    }
  }
  
  func testVectorScale() {
    let a = VectorArrayStorage.create(repeating: Vector3(1, 2, 3), count: 5)
    a.scale(by: 10)
    for i in 0..<5 {
      XCTAssertEqual(a[i], Vector3(10, 20, 30))
    }
  }
  
  func testVectorDot() {
    let a = VectorArrayStorage.create(repeating: Vector3(1, 2, 3), count: 5)
    let b = VectorArrayStorage.create(repeating: Vector3(10, 20, 30), count: 5)
    let dot = b.withUnsafeMutableBufferPointer { bBuffer in
      return a.dot(UnsafeRawBufferPointer(bBuffer))
    }
    XCTAssertEqual(dot, 5 * (Double(1 * 10 + 2 * 20 + 3 * 30)))
  }
  
}
