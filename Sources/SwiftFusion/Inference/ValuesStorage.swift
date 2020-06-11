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
  
  /// Moves `self` along the vector starting at `directionsStart`.
  ///
  /// Precondition: `directionsStart` points to memory with at least `count` initialized
  /// `Element.TangentVector`s where `Element` is the element type of `self`.
  final func move(along directionsStart: UnsafeRawPointer) {
    differentiableImplementation.move_(along: directionsStart)
  }
}

extension AnyArrayBuffer where Storage: AnyDifferentiableStorage {
  /// Moves each element of `self` along the corresponding element of `directions`.
  ///
  /// Precondition: `direction` has at least `count` elements of type `Element.TangentVector`,
  /// where `Element` is the type of `self`.
  mutating func move<DirectionStorage>(along directions: AnyArrayBuffer<DirectionStorage>) {
    ensureUniqueStorage()
    directions.withUnsafeRawPointerToElements { directionsStart in
      storage.move(along: directionsStart)
    }
  }
}

/// Contiguous storage of homogeneous `Differentiable` values of statically unknown type.
protocol AnyDifferentiableStorageImplementation: AnyDifferentiableStorage {
  /// Moves `self` along the vector starting at `directionsStart`.
  ///
  /// Precondition: `directionsStart` points to memory with at least `count` initialized
  /// `Element.TangentVector`s where `Element` is the element type of `self`.
  func move_(along directionsStart: UnsafeRawPointer)
}

/// APIs that depend on `Differentiable` `Element` type.
extension ArrayStorageImplementation where Element: Differentiable {
  /// Moves each element of `self` along the corresponding element of `directions`.
  ///
  /// Precondition: `directions.count >= count`.
  func move<Directions: Collection>(along directions: Directions)
    where Directions.Element == Element.TangentVector
  {
    withUnsafeMutableBufferPointer { b in
      zip(b.indices, directions).forEach { (i, d) in b[i].move(along: d) }
    }
  }
  
  /// Moves `self` along the vector starting at `directionsStart`.
  ///
  /// Precondition: `directionsStart` points to memory with at least `count` initialized
  /// `Element.TangentVector`s where `Element` is the element type of `self`.
  func move_(along directionsStart: UnsafeRawPointer) {
    move(
      along: UnsafeBufferPointer(
        start: directionsStart.assumingMemoryBound(to: Element.TangentVector.self),
        count: count
      )
    )
  }
}

extension ArrayBuffer where Element: Differentiable {
  /// Moves each element of `self` along the corresponding element of `directions`.
  ///
  /// Precondition: `directions.count >= count`.
  mutating func move<Directions: Collection>(along directions: Directions)
    where Directions.Element == Element.TangentVector
  {
    ensureUniqueStorage()
    storage.move(along: directions)
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
  
  /// Adds the vector starting at `otherStart` to `self`.
  ///
  /// Precondition: `otherStart` points to memory with at least `count` initialized `Element`s,
  /// where `Element` is the element type of `self`.
  final func add(_ otherStart: UnsafeRawPointer) {
    vectorImplementation.add_(otherStart)
  }
  
  /// Scales each element of `self` by `scalar`.
  final func scale(by scalar: Double) {
    vectorImplementation.scale_(by: scalar)
  }
  
  /// Returns the dot product of `self` with the vector starting at `otherStart`.
  ///
  /// This is the sum of the dot products of corresponding elements.
  ///
  /// Precondition: `otherStart` points to memory with at least `count` initialized `Element`s,
  /// where `Element` is the element type of `self`.
  final func dot(_ otherStart: UnsafeRawPointer) -> Double {
    vectorImplementation.dot_(otherStart)
  }
}

extension AnyArrayBuffer where Storage: AnyVectorStorage {
  /// Adds `other` to `self`.
  ///
  /// Precondition: `other.count >= count`.
  mutating func add<OtherStorage>(_ other: AnyArrayBuffer<OtherStorage>) {
    ensureUniqueStorage()
    other.withUnsafeRawPointerToElements { otherStart in
      storage.add(otherStart)
    }
  }

  /// Scales each element of `self` by scalar.
  mutating func scale(by scalar: Double) {
    ensureUniqueStorage()
    storage.scale(by: scalar)
  }

  /// Returns the dot product of `self` with `other`.
  ///
  /// Precondition: `other.count >= count`.
  func dot<OtherStorage>(_ other: AnyArrayBuffer<OtherStorage>) -> Double {
    other.withUnsafeRawPointerToElements { otherStart in
      storage.dot(otherStart)
    }
  }
}

/// Contiguous storage of homogeneous `EuclideanVector` values of statically unknown type.
protocol AnyVectorStorageImplementation:
  AnyVectorStorage, AnyDifferentiableStorageImplementation
{
  /// Adds the vector starting at `otherStart` to `self`.
  ///
  /// Precondition: `otherStart` points to memory with at least `count` initialized `Element`s,
  /// where `Element` is the element type of `self`.
  func add_(_ otherStart: UnsafeRawPointer)
  
  /// Scales each element of `self` by `scalar`.
  func scale_(by scalar: Double)
  
  /// Returns the dot product of `self` with the vector starting at `otherStart`.
  ///
  /// This is the sum of the dot products of corresponding elements.
  ///
  /// Precondition: `otherStart` points to memory with at least `count` initialized `Element`s,
  /// where `Element` is the element type of `self`.
  func dot_(_ otherStart: UnsafeRawPointer) -> Double
}

/// APIs that depend on `EuclideanVector` `Element` type.
extension ArrayStorageImplementation where Element: EuclideanVector {
  /// Adds `others` to `self`.
  ///
  /// Precondition: `others.count >= count`.
  func add<Others: Collection>(_ others: Others) where Others.Element == Element {
    withUnsafeMutableBufferPointer { b in
      zip(b.indices, others).forEach { (i, o) in b[i] += o }
    }
  }
  
  /// Returns the dot product of `self` with `others`.
  ///
  /// This is the sum of the dot products of corresponding elements.
  ///
  /// Precondition: `others.count >= count`.
  func dot<Others: Collection>(_ others: Others) -> Double where Others.Element == Element {
    return withUnsafeMutableBufferPointer { b in
      return zip(b, others).lazy.map { (s, o) in s.dot(o)}.reduce(0, +)
    }
  }
  
  /// Adds the vector starting at `otherStart` to `self`.
  ///
  /// Precondition: `otherStart` points to memory with at least `count` initialized `Element`s.
  func add_(_ otherStart: UnsafeRawPointer) {
    add(UnsafeBufferPointer(start: otherStart.assumingMemoryBound(to: Element.self), count: count))
  }
  
  /// Scales each element of `self` by `scalar`.
  func scale_(by scalar: Double) {
    withUnsafeMutableBufferPointer { b in
      b.indices.forEach { i in b[i] *= scalar }
    }
  }
  
  /// Returns the dot product of `self` with the vector starting at `otherStart`.
  ///
  /// This is the sum of the dot products of corresponding elements.
  ///
  /// Precondition: `otherStart` points to memory with at least `count` initialized `Element`s.
  func dot_(_ otherStart: UnsafeRawPointer) -> Double {
    return dot(
      UnsafeBufferPointer(start: otherStart.assumingMemoryBound(to: Element.self), count: count)
    )
  }
}

extension ArrayBuffer where Element: EuclideanVector {
  /// Adds `others` to `self`.
  ///
  /// Precondition: `others.count >= count`.
  mutating func add<Others: Collection>(_ others: Others) where Others.Element == Element {
    ensureUniqueStorage()
    storage.add(others)
  }

  /// Scales each element of `self` by `scalar`.
  mutating func scale(by scalar: Double) {
    ensureUniqueStorage()
    storage.scale_(by: scalar)
  }

  /// Returns the dot product of `self` with `others`.
  ///
  /// This is the sum of the dot products of corresponding elements.
  ///
  /// Precondition: `others.count >= count`.
  func dot<Others: Collection>(_ others: Others) -> Double where Others.Element == Element {
    storage.dot(others)
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
