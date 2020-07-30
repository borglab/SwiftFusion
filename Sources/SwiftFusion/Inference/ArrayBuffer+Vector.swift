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

extension ArrayBuffer: EuclideanVector where Element: EuclideanVector {
  /// Returns a zero tensor shape-compatible with all other tensors
  public static var zero: Self { .init() }
  
  /// Replaces lhs with the product of `lhs` and `rhs`.
  public static func *= (lhs: inout ArrayBuffer, rhs: Double) -> ArrayBuffer {
    if lhs.isEmpty { return }
    if rhs == 0 { lhs = .zero }
    else { lhs.update(elementwiseWith: rhs, *=, *) }
  }

  /// Returns the product of `lhs` and `rhs`.
  public static func * (lhs: ArrayBuffer, rhs: Double) -> ArrayBuffer {
    if lhs.isEmpty { return lhs }
    if rhs == 0 { return .zero }
    return .init(lhs.lazy.map { $0 * rhs })
  }

  /// Returns the dot product of `self` with `other`.
  ///
  /// - Requires: `self.tensorShapeIsCompatible(withThatOf: other)`
  public func dot(other: ArrayBuffer) -> Double {
    if self.isEmpty || other.isEmpty { return 0 }
    assert(self.tensorShapeIsCompatible(withThatOf: other))
    return self.withUnsafeBufferPointer { lhs in
      other.withUnsafeBufferPointer { rhs in
        (0..<lhs.count).reduce(0) { sum, i in sum += lhs[i].dot(rhs[i]) }
      }
    }
  }

  /// Replaces `lhs` with the sum of `lhs` and `rhs`
  ///
  /// - Requires: `lhs.tensorShapeIsCompatible(withThatOf: rhs)`
  public static func += (lhs: inout ArrayBuffer, rhs: ArrayBuffer) {
    if rhs.isEmpty { return }
    else if lhs.isEmpty { lhs = rhs }
    else { lhs.update(elementwiseWith: rhs, +=, +) }
  }

  /// Returns the result of subtracting `rhs` from `lhs`.
  ///
  /// - Requires: `lhs.tensorShapeIsCompatible(withThatOf: rhs)`
  public static func -= (lhs: inout ArrayBuffer, rhs: ArrayBuffer) {
    if rhs.isEmpty { return }
    else { lhs.update(elementwiseWith: rhs, -=, -) }
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

