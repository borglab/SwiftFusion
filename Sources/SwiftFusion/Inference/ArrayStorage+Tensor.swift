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
import PenguinStructures

extension ArrayStorage {
  /// Returns `true` if `other` is compatibly-shaped.
  ///
  /// The empty `ArrayStorage` is considered to be a tensor full of zeros, compatible with any other
  /// size ArrayStorage.
  func tensorShapeIsCompatible<T>(withThatOf other: ArrayStorage<T>) -> Bool {
    return isEmpty || other.isEmpty || self.count == other.count
  }

  /// Creates a new instance containing the elements of `a0` combined with the corresponding
  /// elements of `a1` via calling `combine`.
  ///
  /// - Requires: a0.count == a1.count
  init<E0, E1>(
    elementwise a0: ArrayStorage<E0>, _ a1: ArrayStorage<E1>,
    _ combine: (_ a0x: E0, _ a1x: E1)->Element
  ) {
    assert(a0.count == a1.count, "shape mismatch (\(a0.count) != \(a1.count))")
    self.init(count: a0.count) { target in 
      a0.withUnsafeMutableBufferPointer { b0 in
        a1.withUnsafeMutableBufferPointer { b1 in
          for i in 0..<b0.count {
            (target + i).initialize(to: combine(b0[i], b1[i]))
          }
        }
      }
    }
  }

  /// Creates a new instance containing the elements of `a0` combined with the corresponding
  /// elements of `a1` via calling `combine`.
  ///
  /// - Requires: a0.count == a1.count
  init<E0, T>(
    elementwise a0: ArrayStorage<E0>, _ a1: T,
    _ combine: (_ a0x: E0, _ a1x: T)->Element
  ) {
    self.init(count: a0.count) { target in 
      a0.withUnsafeMutableBufferPointer { b0 in
        for i in 0..<b0.count {
          (target + i).initialize(to: combine(b0[i], a1))
        }
      }
    }
  }

  /// Calls `updateElement(&myX, aX)` for each pair of corresponding elements `myX` and `aX` of
  /// `self` and `a`.
  ///
  /// - Requires: `self.count == a.count`
  func update<E>(
    elementwiseWith a: ArrayStorage<E>,
    _ updateElement: (_ myX: inout Element, _ aX: E)->Void
  ) {
    self.withUnsafeMutableBufferPointer { me in
      a.withUnsafeMutableBufferPointer { b in
        assert(me.count == b.count, "shape mismatch (\(me.count) != \(b.count))")
        for i in 0..<me.count { updateElement(&me[i], b[i]) }
      }
    }
  }

  /// Calls `updateElement(&myX, aX)` for element `myX` of `self`.
  ///
  /// - Requires: `self.count == a.count`
  func update<T>(
    elementwise a: T,
    _ updateElement: (_ myX: inout Element, _ aX: T)->Void
  ) {
    self.withUnsafeMutableBufferPointer { me in
      for i in 0..<me.count { updateElement(&me[i], a) }
    }
  }
}

extension ArrayStorage: Equatable where Element: AdditiveArithmetic {
  /// Returns `true` if `other` is equal to `self`, as a tensor value.
  ///
  /// - Requires: `lhs.shapeIsCompatible(withThatOf: rhs)`
  public static func == (lhs: ArrayStorage, rhs: ArrayStorage) -> Bool {
    if lhs.isEmpty { return rhs.allSatisfy { $0 == .zero } }
    if rhs.isEmpty { return lhs.allSatisfy { $0 == .zero } }
    assert(lhs.count == rhs.count, "shape mismatch (\(lhs.count) != \(rhs.count))")
    return lhs.elementsEqual(rhs)
  }
}

/* DWA TODO: eliminate?
extension ArrayStorage: AdditiveArithmetic where Element: AdditiveArithmetic {
  /// Returns a zero tensor shape-compatible with all other tensors
  public static var zero: Self { .init(minimumCapacity: 0) }
  
  /// Returns the sum of `lhs` and `rhs`.
  ///
  /// - Requires: `lhs.shapeIsCompatible(withThatOf: rhs)`
  public static func + (lhs: ArrayStorage, rhs: ArrayStorage) -> ArrayStorage {
    if lhs.isEmpty { return rhs }
    if rhs.isEmpty { return lhs }
    return .init(elementwise: lhs, rhs, +)
  }

  /// Returns the result of subtracting `rhs` from `lhs`.
  ///
  /// - Requires: `lhs.shapeIsCompatible(withThatOf: rhs)`
  public static func - (lhs: ArrayStorage, rhs: ArrayStorage) -> ArrayStorage {
    if rhs.isEmpty { return lhs }
    return .init(elementwise: lhs, rhs, -)
  }

  /// Replaces `lhs` with the sum of `lhs` and `rhs`
  ///
  /// - Requires: `lhs.shapeIsCompatible(withThatOf: rhs)`
  public static func += (lhs: inout ArrayStorage, rhs: ArrayStorage) {
    if rhs.isEmpty { return }
    else if lhs.isEmpty { lhs = rhs }
    else { lhs.update(elementwiseWith: rhs, +=) }
  }

  /// Returns the result of subtracting `rhs` from `lhs`.
  ///
  /// - Requires: `lhs.shapeIsCompatible(withThatOf: rhs)`
  public static func -= (lhs: inout ArrayStorage, rhs: ArrayStorage) {
    if rhs.isEmpty { return }
    else { lhs.update(elementwiseWith: rhs, -=) }
  }
}
*/
