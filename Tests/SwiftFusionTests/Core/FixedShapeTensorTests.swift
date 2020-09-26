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

import XCTest

import PenguinStructures
import SwiftFusion
import TensorFlow

class FixedShapeTensorTests: XCTestCase {
  /// Test that we can differentiably convert a `Tensor` to a `FixedShapeTensor`.
  func testInitFromTensor() {
    let t = Tensor<Double>(ones: [10, 10])
    let (value, pb) = valueWithPullback(at: t) { Tensor10x10($0) }
    XCTAssertEqual(value, Tensor10x10(t))
    for i in 0..<10 {
      for j in 0..<10 {
        var basisVector = Tensor<Double>(zeros: [10, 10])
        basisVector[i, j] = Tensor(1)
        XCTAssertEqual(pb(Tensor10x10(basisVector)), basisVector)
      }
    }
  }

  /// Test that we can differentiably convert a `FixedShapeTensor` to a `Tensor`.
  func testToTensor() {
    let t = Tensor<Double>(ones: [10, 10])
    let fst = Tensor10x10(t)
    let (value, pb) = valueWithPullback(at: fst) { $0.tensor }
    XCTAssertEqual(value, t)
    for i in 0..<10 {
      for j in 0..<10 {
        var basisVector = Tensor<Double>(zeros: [10, 10])
        basisVector[i, j] = Tensor(1)
        XCTAssertEqual(pb(basisVector), Tensor10x10(basisVector))
      }
    }
  }

  func testVectorConformance() {
    let s = (0..<100).lazy.map { Double($0) }
    let v = Tensor10x10(Tensor(rangeFrom: 0, to: 100, stride: 1).reshaped(to: Tensor10x10.shape))
    v.checkVectorSemantics(
      expectingScalars: s,
      writingScalars: (100..<200).lazy.map { Double($0) })
    v.scalars.checkRandomAccessCollectionSemantics(expecting: s)
  }
}
