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

extension ArrayBuffer/*: EuclideanVector*/ where Element: EuclideanVector {
  /// Replaces lhs with the product of `lhs` and `rhs`.
  public static func *= (lhs: inout ArrayBuffer, rhs: Double) -> Void {
    if lhs.isEmpty { return }
    if rhs == 0 { lhs = .zero }
    else {
      lhs.update(elementwise: rhs, *=, { l, r in r * l })
    }
  }

  /// Returns the product of `lhs` and `rhs`.
  public static func * (lhs: ArrayBuffer, rhs: Double) -> ArrayBuffer {
    if lhs.isEmpty { return lhs }
    if rhs == 0 { return .zero }
    return .init(lhs.lazy.map { rhs * $0 })
  }

  /// Returns the dot product of `self` with `other`.
  ///
  /// - Requires: `self.tensorShapeIsCompatible(withThatOf: other)`
  public func dot(other: ArrayBuffer) -> Double {
    if self.isEmpty || other.isEmpty { return 0 }
    assert(self.tensorShapeIsCompatible(withThatOf: other))
    return self.withUnsafeBufferPointer { lhs in
      other.withUnsafeBufferPointer { rhs in
        (0..<lhs.count).reduce(0) { sum, i in sum + lhs[i].dot(rhs[i]) }
      }
    }
  }

  /// Creates an instance whose elements are `scalars`.
  ///
  /// Precondition: `scalars` must have an element count that `Self` can hold (e.g. if `Self` is a
  /// fixed-size vectors, then `scalars` must have exactly the right number of elements).
  // TODO: remove?
  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    fatalError("can't implement this")
  }
}

