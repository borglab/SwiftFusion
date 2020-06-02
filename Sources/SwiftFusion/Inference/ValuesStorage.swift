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

/// Contiguous storage for factor graph values (variable assignments, error vectors) of statically
/// unknown type.

import PenguinStructures

// MARK: - Storage for contiguous arrays of `Differentiable` values.

/// Contiguous storage of homogeneous `Differentiable` values of statically unknown type.
class AnyDifferentiableStorage: AnyArrayStorage {
  typealias DifferentiableImplementation = AnyDifferentiableStorageImplementation
  var differentiableImplementation: DifferentiableImplementation {
    fatalError("implement me!")
  }
  
  /// Moves each element of `self` along the corresponding element of `direction`.
  ///
  /// `direction` must point to a buffer of `Element.TangentVector`s, where `Element` is the type
  /// stored in `self`, and `direction.count` must be `self.count`.
  final func move(along direction: UnsafeRawBufferPointer) {
    differentiableImplementation.move_(along: direction)
  }
}

/// Contiguous storage of homogeneous `Differentiable` values of statically unknown type.
protocol AnyDifferentiableStorageImplementation: AnyDifferentiableStorage {
  /// Moves each element of `self` along the corresponding element of `direction`.
  ///
  /// `direction` must point to a buffer of `Element.TangentVector`s, where `Element` is the type
  /// stored in `self`, and `direction.count` must be `self.count`.
  func move_(along direction: UnsafeRawBufferPointer)
}

/// APIs that depend on `Differentiable` `Element` type.
extension ArrayStorageImplementation where Element: Differentiable {
  /// Moves each element of `self` along the corresponding element of `direction`.
  ///
  /// `direction.count` must be `self.count`.
  func move(along direction: UnsafeBufferPointer<Element.TangentVector>) {
    withUnsafeMutableBufferPointer { elementBuffer in
      precondition(elementBuffer.count == direction.count)
      var elementIndex = elementBuffer.startIndex
      var directionIndex = direction.startIndex
      while elementIndex != elementBuffer.endIndex {
        elementBuffer[elementIndex].move(along: direction[directionIndex])
        elementIndex = elementBuffer.index(after: elementIndex)
        directionIndex = direction.index(after: directionIndex)
      }
    }
  }
  
  /// Moves each element of `self` along the corresponding element of `direction`.
  ///
  /// `direction` must point to a buffer of `Element.TangentVector`s, and  `direction.count` must
  /// be `self.count`.
  func move_(along direction: UnsafeRawBufferPointer) {
    move(along: UnsafeBufferPointer(
      start: direction.baseAddress?.assumingMemoryBound(to: Element.TangentVector.self),
      count: direction.count / MemoryLayout<Element.TangentVector>.stride
    ))
  }
}

/// Type-erasable storage for contiguous `Differentiable` `Element` instances.
///
/// Note: instances have reference semantics.
final class DifferentiableArrayStorage<Element: Differentiable>:
  AnyDifferentiableStorage, AnyDifferentiableStorageImplementation,
  ArrayStorageImplementation
{
  override var implementation: AnyArrayStorageImplementation { self }
  override var differentiableImplementation: AnyDifferentiableStorageImplementation {
    self
  }
}

// MARK: - Storage for contiguous arrays of `EuclideanVector` values.

/// Contiguous storage of homogeneous `EuclideanVector` values of statically unknown type.
class AnyVectorStorage: AnyDifferentiableStorage {
  typealias VectorImplementation = AnyVectorStorageImplementation
  var vectorImplementation: VectorImplementation {
    fatalError("implement me!")
  }
  
  /// Adds each element of `other` to the corresponding element of `self`.
  ///
  /// `other` must point to a buffer of the same type stored in `self`, and `other.count` must be
  /// `self.count`.
  final func add(_ other: UnsafeRawBufferPointer) {
    vectorImplementation.add_(other)
  }
  
  /// Scales each element of `self` by `scalar`.
  final func scale(by scalar: Double) {
    vectorImplementation.scale_(by: scalar)
  }
  
  /// Returns the dot product of `self` with `other`.
  ///
  /// This is the sum of the dot product of corresponding elements.
  ///
  /// `other` must point to a buffer of the same type stored in `self`, and `other.count` must be
  /// `self.count`.
  final func dot(_ other: UnsafeRawBufferPointer) -> Double {
    vectorImplementation.dot_(other)
  }
}

/// Contiguous storage of homogeneous `EuclideanVector` values of statically unknown type.
protocol AnyVectorStorageImplementation:
  AnyVectorStorage, AnyDifferentiableStorageImplementation
{
  /// Adds each element of `other` to the corresponding element of `self`.
  ///
  /// `other` must point to a buffer of the same type stored in `self`, and `other.count` must be
  /// `self.count`.
  func add_(_ other: UnsafeRawBufferPointer)
  
  /// Scales each element of `self` by `scalar`.
  func scale_(by scalar: Double)
  
  /// Returns the dot product of `self` with `other`.
  ///
  /// This is the sum of the dot products of corresponding elements.
  ///
  /// `other` must point to a buffer of the same type stored in `self`, and `other.count` must be
  /// `self.count`.
  func dot_(_ other: UnsafeRawBufferPointer) -> Double
}

/// APIs that depend on `EuclideanVector` `Element` type.
extension ArrayStorageImplementation where Element: EuclideanVector {
  /// Adds each element of `other` to the corresponding element of `self`.
  ///
  /// `other.count` must be `self.count`.
  func add(_ other: UnsafeBufferPointer<Element>) {
    withUnsafeMutableBufferPointer { elementBuffer in
      precondition(elementBuffer.count == other.count)
      var elementIndex = elementBuffer.startIndex
      var otherIndex = other.startIndex
      while elementIndex != elementBuffer.endIndex {
        elementBuffer[elementIndex] += other[otherIndex]
        elementIndex = elementBuffer.index(after: elementIndex)
        otherIndex = other.index(after: otherIndex)
      }
    }
  }
  
  /// Returns the dot product of `self` with `other`.
  ///
  /// This is the sum of the dot products of corresponding elements.
  ///
  /// `other.count` must be `self.count`.
  func dot(_ other: UnsafeBufferPointer<Element>) -> Double {
    var result = Double(0)
    withUnsafeMutableBufferPointer { elementBuffer in
      precondition(elementBuffer.count == other.count)
      var elementIndex = elementBuffer.startIndex
      var otherIndex = other.startIndex
      while elementIndex != elementBuffer.endIndex {
        result += elementBuffer[elementIndex].dot(other[otherIndex])
        elementIndex = elementBuffer.index(after: elementIndex)
        otherIndex = other.index(after: otherIndex)
      }
    }
    return result
  }
  
  /// Adds each element of `other` to the corresponding element of `self`.
  ///
  /// `other` must point to a buffer of the same type stored in `self`, and `other.count` must be
  /// `self.count`.
  func add_(_ other: UnsafeRawBufferPointer) {
    add(UnsafeBufferPointer(
      start: other.baseAddress?.assumingMemoryBound(to: Element.self),
      count: other.count / MemoryLayout<Element>.stride
    ))
  }
  
  /// Scales each element of `self` by `scalar`.
  func scale_(by scalar: Double) {
    withUnsafeMutableBufferPointer { elementBuffer in
      for index in elementBuffer.indices {
        elementBuffer[index] *= scalar
      }
    }
  }
  
  /// Returns the dot product of `self` with `other`.
  ///
  /// This is the sum of the dot products of corresponding elements.
  ///
  /// `other` must point to a buffer of the same type stored in `self`, and `other.count` must be
  /// `self.count`.
  func dot_(_ other: UnsafeRawBufferPointer) -> Double {
    return dot(UnsafeBufferPointer(
      start: other.baseAddress?.assumingMemoryBound(to: Element.self),
      count: other.count / MemoryLayout<Element>.stride
    ))
  }
}

/// Type-erasable storage for contiguous `EuclideanVector` `Element` instances.
///
/// Note: instances have reference semantics.
final class VectorArrayStorage<Element: EuclideanVector>:
  AnyVectorStorage, AnyVectorStorageImplementation,
  ArrayStorageImplementation
{  
  override var implementation: AnyArrayStorageImplementation { self }
  override var differentiableImplementation: DifferentiableImplementation { self }
  override var vectorImplementation: AnyVectorStorageImplementation { self }
}
