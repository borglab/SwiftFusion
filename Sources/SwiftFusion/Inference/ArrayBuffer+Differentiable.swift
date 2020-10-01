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
import PenguinStructures

extension ArrayBuffer: Differentiable where Element: Differentiable {
  public typealias TangentVector = ArrayBuffer<Element.TangentVector>

  /// Returns the zero `TangentVector`s of the contained elements.
   // public var tangentVectorZeros: ArrayBuffer<Element.TangentVector> { .init() }

  /// Moves each element of `self` along the corresponding element of `directions`.
  ///
  /// - Requires: `directions.count == self.count`.
  public mutating func move(along directions: TangentVector) {
    if directions.isEmpty { return }
    update(elementwiseWith: directions, { $0.move(along: $1) })
  }

  /// A function returning the zero `TangentVector`s of the contained elements.
  @noDerivative
  public var zeroTangentVectorInitializer: () -> TangentVector {
    { .zero }
  }
}

extension ArrayBuffer where Element: Differentiable {
  // DWA TODO: replace this with the use of zeroTangentVectorInitializer
  /// Returns the zero `TangentVector`s of the contained elements.
  var tangentVectorZeros: ArrayBuffer<Element.TangentVector> {
    withUnsafeBufferPointer { vs in
      .init(vs.lazy.map { $0.zeroTangentVector })
    }
  }
}

