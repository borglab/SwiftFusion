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
  mutating func move(along directionsStart: UnsafeRawPointer) {
    withMutableStorage { s in s.move(along: directionsStart) }
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
  /// Moves `self` along the vector starting at `directionsStart`.
  ///
  /// Precondition: all values in `directionsStart..<directionsStart+count` point to initialized
  /// `Element.TangentVector`s.
  func move(along directionsStart: UnsafePointer<Element.TangentVector>) {
    withUnsafeMutableBufferPointer { b in
      b.indices.forEach { i in b[i].move(along: directionsStart[i]) }
    }
  }
  
  /// Moves `self` along the vector starting at `directionsStart`.
  ///
  /// Precondition: `directionsStart` points to memory with at least `count` initialized
  /// `Element.TangentVector`s where `Element` is the element type of `self`.
  func move_(along directionsStart: UnsafeRawPointer) {
    move(along: directionsStart.assumingMemoryBound(to: Element.TangentVector.self))
  }
}

extension ArrayBuffer where Element: Differentiable {
  mutating func move(along directionsStart: UnsafePointer<Element.TangentVector>) {
    withMutableStorage { s in s.move(along: directionsStart) }
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
  final func dot(_ other: UnsafeRawPointer) -> Double {
    vectorImplementation.dot_(other)
  }
}

extension AnyArrayBuffer where Storage: AnyVectorStorage {
  mutating func add(_ otherStart: UnsafeRawPointer) {
    withMutableStorage { s in s.add(otherStart) }
  }

  mutating func scale(by scalar: Double) {
    withMutableStorage { s in s.scale(by: scalar) }
  }

  func dot(_ other: UnsafeRawPointer) -> Double {
    withStorage { s in s.dot(other) }
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
  /// Adds the vector starting at `otherStart` to `self`.
  ///
  /// Precondition: all values in `otherStart..<otherStart+count` refer to initialized `Element`s.
  func add(_ otherStart: UnsafePointer<Element>) {
    withUnsafeMutableBufferPointer { b in
      b.indices.forEach { i in b[i] += otherStart[i] }
    }
  }
  
  /// Returns the dot product of `self` with the vector starting at `otherStart`.
  ///
  /// This is the sum of the dot products of corresponding elements.
  ///
  /// Precondition: all values in `otherStart..<otherStart+count` refer to initialized `Element`s.
  func dot(_ otherStart: UnsafePointer<Element>) -> Double {
    return withUnsafeMutableBufferPointer { b in
      return b.enumerated().lazy.map { (n, v) in v.dot(otherStart[n])}.reduce(0, +)
    }
  }
  
  /// Adds the vector starting at `otherStart` to `self`.
  ///
  /// Precondition: `otherStart` points to memory with at least `count` initialized `Element`s.
  func add_(_ otherStart: UnsafeRawPointer) {
    add(otherStart.assumingMemoryBound(to: Element.self))
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
    return dot(otherStart.assumingMemoryBound(to: Element.self))
  }
}

extension ArrayBuffer where Element: EuclideanVector {
  mutating func add(_ otherStart: UnsafePointer<Element>) {
    withMutableStorage { s in s.add(otherStart) }
  }

  mutating func scale(by scalar: Double) {
    withMutableStorage { s in s.scale_(by: scalar) }
  }

  func dot(_ otherStart: UnsafePointer<Element>) -> Double {
    withStorage { s in s.dot(otherStart) }
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
