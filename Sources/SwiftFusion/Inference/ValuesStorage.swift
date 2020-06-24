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

// MARK: - Algorithms on arrays of `Differentiable` values.

extension ArrayStorage where Element: Differentiable, Element.TangentVector: EuclideanVectorN {
  /// Returns the zero `TangentVector`s of the contained elements.
  var tangentVectorZeros: ArrayBuffer<Element.TangentVector> {
    withUnsafeMutableBufferPointer { vs in
      .init(vs.lazy.map { $0.zeroTangentVector })
    }
  }

  /// Moves each element of `self` along the corresponding element of `directions`.
  ///
  /// - Requires: `directions.count == self.count`.
  func move(along directions: ArrayBuffer<Element.TangentVector>) {
    assert(directions.count == self.count)
    withUnsafeMutableBufferPointer { vs in
      directions.withUnsafeBufferPointer { ds in
        for i in vs.indices {
          vs[i].move(along: ds[i])
        }
      }
    }
  }
}

// MARK: - Algorithms on arrays of `EuclideanVectorN` values.

extension ArrayStorage where Element: EuclideanVectorN {
  /// Adds `others` to `self`.
  ///
  /// - Requires: `others.count == count`.
  func add(_ others: ArrayBuffer<Element>) {
    assert(others.count == self.count)
    withUnsafeMutableBufferPointer { vs in
      others.withUnsafeBufferPointer { os in
        for i in vs.indices {
          vs[i] += os[i]
        }
      }
    }
  }

  /// Scales each element of `self` by `scalar`.
  func scale(by scalar: Double) {
    withUnsafeMutableBufferPointer { vs in
      for i in vs.indices {
        vs[i] *= scalar
      }
    }
  }

  /// Returns the dot product of `self` with `others`.
  ///
  /// This is the sum of the dot products of corresponding elements.
  ///
  /// - Requires: `others.count == count`.
  func dot(_ others: ArrayBuffer<Element>) -> Double {
    assert(others.count == self.count)
    return withUnsafeMutableBufferPointer { vs in
      others.withUnsafeBufferPointer { os in
        vs.indices.reduce(into: 0) { (result, i) in result += vs[i].dot(os[i]) }
      }
    }
  }

  /// Returns Jacobians that scale each element by `scalar`.
  func jacobians(scalar: Double) -> AnyGaussianFactorArrayBuffer {
    AnyGaussianFactorArrayBuffer(ArrayBuffer(enumerated().lazy.map { (i, _) in
      ScalarJacobianFactor(edges: Tuple1(TypedID<Element, Int>(i)), scalar: scalar)
    }))
  }
}

// MARK: - Type-erased arrays of `Differentiable` values.

typealias AnyDifferentiableArrayBuffer = AnyArrayBuffer<DifferentiableArrayDispatch>

/// An `AnyArrayBuffer` dispatcher that provides algorithm implementations for `Differentiable`
/// elements.
class DifferentiableArrayDispatch: AnyElementDispatch {
  /// The `TangentVector` type of the elements.
  class var tangentVectorType: Any.Type {
    fatalError("implement as in DifferentiableArrayDispatch_")
  }

  /// Returns the zero `TangentVector`s for each of the elements in the `ArrayStorage` whose
  /// address is `storage`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage` whose `Element` has a
  ///   subclass-specific `Differentiable` type.
  class func tangentVectorZeros(_ storage: UnsafeRawPointer)
    -> AnyArrayBuffer<VectorArrayDispatch>
  {
    fatalError("implement as in DifferentiableArrayDispatch_")
  }

  /// Moves each element in the `ArrayStorage` whose address is `storage` along the corresponding
  /// element of `directions`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage` whose `Element`
  ///   has a subclass-specific `Differentiable` type.
  /// - Requires: `directions.elementType == Element.TangentVector.self`.
  class func move(_ storage: UnsafeRawPointer, along directions: AnyElementArrayBuffer) {
    fatalError("implement as in DifferentiableArrayDispatch_")
  }
}

/// An `AnyArrayBuffer` dispatcher that provides algorithm implementations for a
/// specific `Differentiable` element type.
class DifferentiableArrayDispatch_<Element: Differentiable>
  : DifferentiableArrayDispatch, AnyArrayDispatch
  where Element.TangentVector: EuclideanVectorN
{
  /// The `TangentVector` type of the elements.
  override class var tangentVectorType: Any.Type {
    Element.TangentVector.self
  }

  /// Returns the zero `TangentVector`s for each of the elements in the `ArrayStorage` whose
  /// address is `storage`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage<Element>`.
  override class func tangentVectorZeros(_ storage: UnsafeRawPointer)
    -> AnyArrayBuffer<VectorArrayDispatch>
  {
    .init(asStorage(storage).tangentVectorZeros)
  }

  /// Moves each element in the `ArrayStorage` whose address is `storage` along the corresponding
  /// element of `directions`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage<Element>`.
  /// - Requires: `directions.elementType == Element.TangentVector.self`.
  override class func move(_ storage: UnsafeRawPointer, along directions: AnyElementArrayBuffer) {
    asStorage(storage).move(along: .init(unsafelyDowncasting: directions))
  }
}

extension AnyArrayBuffer where Dispatch == DifferentiableArrayDispatch {
  /// Creates an instance from a typed buffer of `Element`
  init<Element: Differentiable>(_ src: ArrayBuffer<Element>)
    where Element.TangentVector: EuclideanVectorN
  {
    self.init(
      storage: src.storage,
      dispatch: DifferentiableArrayDispatch_<Element>.self)
  }
}

extension AnyArrayBuffer where Dispatch: DifferentiableArrayDispatch {
  /// The type of the contained elements' `TangentVector`.
  var tangentVectorType: Any.Type {
    dispatch.tangentVectorType
  }

  /// Returns the zero `TangentVector`s of the contained elements.
  var tangentVectorZeros: AnyArrayBuffer<VectorArrayDispatch> {
    withUnsafePointer(to: storage) { dispatch.tangentVectorZeros($0) }
  }

  /// Moves each element along the corresponding element of `directions`.
  mutating func move(along directions: AnyElementArrayBuffer) {
    ensureUniqueStorage()
    withUnsafePointer(to: storage) { dispatch.move($0, along: directions) }
  }
}

// MARK: - Type-erased arrays of `EuclideanVectorN` values.

typealias AnyVectorArrayBuffer = AnyArrayBuffer<VectorArrayDispatch>

/// An `AnyArrayBuffer` dispatcher that provides algorithm implementations for `EuclideanVectorN`
/// elements.
class VectorArrayDispatch: DifferentiableArrayDispatch {
  /// Adds `others` to the `ArrayStorage` whose address is `storage`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage` whose `Element` has a
  ///   subclass-specific `EuclideanVectorN` type.
  /// - Requires: `others.elementType == Element`.
  /// - Requires: `others.count` is at least the count of the `ArrayStorage` at `storage`.
  class func add(_ storage: UnsafeRawPointer, _ others: AnyElementArrayBuffer) {
    fatalError("implement as in VectorArrayDispatch_")
  }

  /// Scales each element of the `ArrayStorage` whose address is `storage` by `scalar`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage` whose `Element` has a
  ///   subclass-specific `EuclideanVectorN` type.
  class func scale(_ storage: UnsafeRawPointer, by scalar: Double) {
    fatalError("implement as in VectorArrayDispatch_")
  }

  /// Returns the dot product of the `ArrayStorage` whose address is `storage` with `others`.
  ///
  /// This is the sum of the dot products of corresponding elements.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage` whose `Element` has a
  ///   subclass-specific `EuclideanVectorN` type.
  /// - Requires: `others.elementType == Element`.
  /// - Requires: `others.count` is at least the count of the `ArrayStorage` at `storage`.
  class func dot(_ storage: UnsafeRawPointer, _ others: AnyElementArrayBuffer) -> Double {
    fatalError("implement as in VectorArrayDispatch_")
  }

  /// Returns Jacobians that scale each element of `storage` by `scalar`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage` whose `Element` has a
  ///   subclass-specific `EuclideanVectorN` type.
  class func jacobians(_ storage: UnsafeRawPointer, scalar: Double)
    -> AnyGaussianFactorArrayBuffer
  {
    fatalError("implement as in VectorArrayDispatch_")
  }

  /// The type of a `ScalarJacobianFactor` that scales the elements.
  class var scalarJacobianType: Any.Type {
    fatalError("implement as in VectorArrayDispatch_")
  }
}

/// An `AnyArrayBuffer` dispatcher that provides algorithm implementations for a
/// specific `Vector` element type.
class VectorArrayDispatch_<Element: EuclideanVectorN>
  : VectorArrayDispatch, AnyArrayDispatch
{
  /// The `TangentVector` type of the elements.
  override class var tangentVectorType: Any.Type {
    Element.TangentVector.self
  }

  /// Returns the zero `TangentVector`s for each of the elements in the `ArrayStorage` whose
  /// address is `storage`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage<Element>`.
  override class func tangentVectorZeros(_ storage: UnsafeRawPointer)
    -> AnyArrayBuffer<VectorArrayDispatch>
  {
    .init(asStorage(storage).tangentVectorZeros)
  }

  /// Moves each element in the `ArrayStorage` whose address is `storage` along the corresponding
  /// element of `directions`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage<Element>`.
  /// - Requires: `directions.elementType == Element.TangentVector.self`.
  override class func move(_ storage: UnsafeRawPointer, along directions: AnyElementArrayBuffer) {
    asStorage(storage).move(along: .init(unsafelyDowncasting: directions))
  }

  /// Adds `others` to the `ArrayStorage` whose address is `storage`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage<Element>`.
  /// - Requires: `others.elementType == Element`.
  /// - Requires: `others.count` is at least the count of the `ArrayStorage` at `storage`.
  @_specialize(where Element == Vector1)
  @_specialize(where Element == Vector2)
  @_specialize(where Element == Vector3)
  @_specialize(where Element == Vector4)
  @_specialize(where Element == Vector5)
  @_specialize(where Element == Vector6)
  @_specialize(where Element == Vector7)
  @_specialize(where Element == Vector8)
  @_specialize(where Element == Vector9)
  @_specialize(where Element == Vector10)
  @_specialize(where Element == Vector11)
  @_specialize(where Element == Vector12)
  override class func add(_ storage: UnsafeRawPointer, _ others: AnyElementArrayBuffer) {
    asStorage(storage).add(.init(unsafelyDowncasting: others))
  }

  /// Scales each element of the `ArrayStorage` whose address is `storage` by `scalar`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage<Element>`.
  @_specialize(where Element == Vector1)
  @_specialize(where Element == Vector2)
  @_specialize(where Element == Vector3)
  @_specialize(where Element == Vector4)
  @_specialize(where Element == Vector5)
  @_specialize(where Element == Vector6)
  @_specialize(where Element == Vector7)
  @_specialize(where Element == Vector8)
  @_specialize(where Element == Vector9)
  @_specialize(where Element == Vector10)
  @_specialize(where Element == Vector11)
  @_specialize(where Element == Vector12)
  override class func scale(_ storage: UnsafeRawPointer, by scalar: Double) {
    asStorage(storage).scale(by: scalar)
  }

  /// Returns the dot product of the `ArrayStorage` whose address is `storage` with `others`.
  ///
  /// This is the sum of the dot products of corresponding elements.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage<Element>`.
  /// - Requires: `others.elementType == Element`.
  /// - Requires: `others.count` is at least the count of the `ArrayStorage` at `storage`.
  @_specialize(where Element == Vector1)
  @_specialize(where Element == Vector2)
  @_specialize(where Element == Vector3)
  @_specialize(where Element == Vector4)
  @_specialize(where Element == Vector5)
  @_specialize(where Element == Vector6)
  @_specialize(where Element == Vector7)
  @_specialize(where Element == Vector8)
  @_specialize(where Element == Vector9)
  @_specialize(where Element == Vector10)
  @_specialize(where Element == Vector11)
  @_specialize(where Element == Vector12)
  override class func dot(_ storage: UnsafeRawPointer, _ others: AnyElementArrayBuffer) -> Double {
    asStorage(storage).dot(.init(unsafelyDowncasting: others))
  }

  /// Returns Jacobians that scale each element by `scalar`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage<Element>`.
  override class func jacobians(_ storage: UnsafeRawPointer, scalar: Double)
    -> AnyGaussianFactorArrayBuffer
  {
    asStorage(storage).jacobians(scalar: scalar)
  }

  /// The type of a `ScalarJacobianFactor` that scales the elements.
  override class var scalarJacobianType: Any.Type {
    ScalarJacobianFactor<Element>.self
  }
}

extension AnyArrayBuffer where Dispatch == VectorArrayDispatch {
  /// Creates an instance from a typed buffer of `Element`
  init<Element: EuclideanVectorN>(_ src: ArrayBuffer<Element>) {
    self.init(
      storage: src.storage,
      dispatch: VectorArrayDispatch_<Element>.self)
  }
}

extension AnyArrayBuffer where Dispatch: VectorArrayDispatch {
  /// Adds `others` to `self`.
  ///
  /// - Requires: `others.count == count`.
  mutating func add(_ others: AnyElementArrayBuffer) {
    ensureUniqueStorage()
    withUnsafePointer(to: storage) { dispatch.add($0, others) }
  }

  /// Scales each element of `self` by `scalar`.
  mutating func scale(by scalar: Double) {
    ensureUniqueStorage()
    withUnsafePointer(to: storage) { dispatch.scale($0, by: scalar) }
  }

  /// Returns the dot product of `self` with `others`.
  ///
  /// This is the sum of the dot products of corresponding elements.
  ///
  /// - Requires: `others.count == count`.
  func dot(_ others: AnyElementArrayBuffer) -> Double {
    withUnsafePointer(to: storage) { dispatch.dot($0, others) }
  }

  func jacobians(scalar: Double) -> AnyGaussianFactorArrayBuffer {
    withUnsafePointer(to: storage) { dispatch.jacobians($0, scalar: scalar) }
  }

  var scalarJacobianType: Any.Type {
    dispatch.scalarJacobianType
  }
}
