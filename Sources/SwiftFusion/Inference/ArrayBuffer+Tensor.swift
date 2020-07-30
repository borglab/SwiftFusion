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

extension ArrayBuffer {
  /// Returns `true` if `other` is compatibly-shaped.
  ///
  /// The empty `ArrayBuffer` is considered to be a tensor full of zeros, compatible with any other
  /// size ArrayBuffer.
  func tensorShapeIsCompatible<T>(withThatOf other: ArrayBuffer<T>) -> Bool {
    return storage.tensorShapeIsCompatible(withThatOf: other.storage)
  }

  /// Creates a new instance containing the elements of `a0` combined with the corresponding
  /// elements of `a1` via calling `combine`.
  ///
  /// - Requires: a0.count == a1.count
  init<E0, E1>(
    elementwise a0: ArrayBuffer<E0>, _ a1: ArrayBuffer<E1>,
    _ combine: (_ a0x: E0, _ a1x: E1)->Element
  ) {
    self.init(ArrayStorage(elementwise: a0.storage, a1.storage, combine))
  }

  /// Calls `updateElement(&myX, aX)` on each pair of corresponding elements `myX` and `aX` of
  /// `self` and `a`, using `combine` (if supplied) to initialize new storage when required.
  ///
  /// - Requires: `self.count == a.count`
  mutating func update<E>(
    elementwiseWith a: ArrayBuffer<E>,
    _ updateElement: (_ myX: inout Element, _ aX: E)->Void,
    _ combine: Optional<(_ a0x: Element, _ a1x: E)->Element> = nil
  ) {
    while true {
      if isKnownUniquelyReferenced(&storage) {
        storage.update(elementwiseWith: a.storage, updateElement)
        return
      }
      else if let f = combine {
        storage = .init(elementwise: storage, a.storage, f)
        return
      }
      else {
        storage = storage.makeCopy()
      }
    }
  }
}

extension ArrayBuffer: Equatable where Element: AdditiveArithmetic {
  /// Returns `true` if `other` is equal to `self`, as a tensor value.
  ///
  /// - Requires: `lhs.tensorShapeIsCompatible(withThatOf: rhs)`
  public static func == (lhs: ArrayBuffer, rhs: ArrayBuffer) -> Bool {
    return lhs.storage == rhs.storage
  }
}

extension ArrayBuffer: AdditiveArithmetic where Element: AdditiveArithmetic {
  /// Returns a zero tensor shape-compatible with all other tensors
  public static var zero: Self { .init() }
  
  /// Returns the sum of `lhs` and `rhs`.
  ///
  /// - Requires: `lhs.tensorShapeIsCompatible(withThatOf: rhs)`
  public static func + (lhs: ArrayBuffer, rhs: ArrayBuffer) -> ArrayBuffer {
    .init(lhs.storage + rhs.storage)
  }

  /// Returns the result of subtracting `rhs` from `lhs`.
  ///
  /// - Requires: `lhs.tensorShapeIsCompatible(withThatOf: rhs)`
  public static func - (lhs: ArrayBuffer, rhs: ArrayBuffer) -> ArrayBuffer {
    .init(lhs.storage - rhs.storage)
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
}

