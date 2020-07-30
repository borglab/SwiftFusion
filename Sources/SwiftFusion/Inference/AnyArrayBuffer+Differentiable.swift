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

/*
import PenguinStructures

fileprivate typealias AnyDifferentiableArrayBuffer = AnyArrayBuffer<DifferentiableArrayDispatch>

/// An `AnyArrayBuffer` dispatcher that provides algorithm implementations for `Differentiable`
/// elements.
fileprivate class DifferentiableArrayDispatch {
  /// The `TangentVector` type of the elements.
  // TODO: eliminate?
  final let tangentVectorType: Any.Type

  /// A function returning the zero `TangentVector`s for each of the elements in `self`
  final let tangentVectorZeros: (AnyDifferentiableArrayBuffer) -> AnyVectorArrayBuffer

  /// A function that moves each element in the `ArrayStorage` whose address is `storage` along the
  /// corresponding element of `directions`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage` whose `Element`
  ///   has a subclass-specific `Differentiable` type.
  /// - Requires: `directions.elementType == Element.TangentVector.self`.
  final let move: (inout AnyDifferentiableArrayBuffer, _ directions: AnyVectorArrayBuffer) -> Void

  /// Creates an instance for elements of type `Element`.
  init<Element: Differentiable>(_: Type<Element>)
    where Element.TangentVector: EuclideanVector
  {
    tangentVectorType = Element.TangentVector.self
    // TODO: try existingElementType instead of unsafelyAssumingElementType.
    tangentVectorZeros = {
      $0[unsafelyAssumingElementType: Type<Element>()].tangentVectorZeros
    }
    move = {
      $0[unsafelyAssumingElementType: Type<Element>()].move(
        along: $1[unsafelyAssumingElementType: Type<Element.TangentVector>()]
    }
  }
}

// MARK: - Type-erased arrays of `EuclideanVectorN` values.

fileprivate typealias AnyVectorArrayBuffer = AnyArrayBuffer<VectorArrayDispatch>

/// An `AnyArrayBuffer` dispatcher that provides algorithm implementations for `EuclideanVectorN`
/// elements.
///
/// For the purposes of equality and other mathematical operations, empty `ArrayStorage` is treated
/// as a ragged tensor of zeros, whose shape is compatible with that of any other `ArrayStorage`.
/// Two non-empty `ArrayStorage` instances have compatible shape iff they have the same `count`.
///
/// All function-typed properties have the preconditions that:
///
/// - The `storage` parameter points to an `AnyArrayStorage` whose `elementType` matches the
///   `Element` type passed to `init`, having a shape compatible with any `others` parameter.
/// - Any `others` parameter has an `elementType` matching the `Element` type passed to `init`.
fileprivate class VectorArrayDispatch: DifferentiableArrayDispatch {
  typealias VectorBuffer = AnyVectorArrayBuffer
  
  /// A function returning `true` iff `others` is equal to the `ArrayStorage` whose address is
  /// `storage`.
  final let equals: (_ lhs: VectorBuffer, _ rhs: VectorBuffer) -> Bool

  final let sum: (_ lhs: VectorBuffer, _ rhs: VectorBuffer) -> VectorBuffer
  final let add: (_ lhs: inout VectorBuffer, _ rhs: VectorBuffer) -> Void 

  final let difference: (_ lhs: VectorBuffer, _ rhs: VectorBuffer) -> VectorBuffer
  final let subtract: (_ lhs: inout VectorBuffer, _ rhs: VectorBuffer) -> Void 

  /*
  /// A function that scales each element of the `ArrayStorage` whose address is `storage` by
  /// `factor`.
  final let scale: (_ target: inout VectorBuffer, _ factor: Double) -> Void

  /// A function returning the dot product of the `ArrayStorage` whose address is `storage` with
  /// `others`.
  ///
  /// - Note: the result is the sum of the dot products of corresponding elements.
  final let dot: (_ lhs: VectorBuffer, _ rhs: VectorBuffer) -> Double

  /// A function returning Jacobians that scale each element of `storage` by `scalar`.
  final let jacobians:
    (_ target: VectorBuffer, _ factor: Double) -> AnyGaussianFactorArrayBuffer
*/
  /// Creates an instance for elements of type `Element`.
  init<Element: EuclideanVector>(_: Type<Element>, _: () = ()) {
    equals = { lhs, rhs in
      lhs[unsafelyAssumingElementType: Type<Element>()]
        == rhs[unsafelyAssumingElementType: Type<Element>()]
    }
    sum = { lhs, rhs in
      lhs[unsafelyAssumingElementType: Type<Element>()]
      + rhs[unsafelyAssumingElementType: Type<Element>()]
    }
    add = { lhs, rhs in
      lhs[unsafelyAssumingElementType: Type<Element>()]
      += rhs[unsafelyAssumingElementType: Type<Element>()]
    }
    difference = { lhs, rhs in
      lhs[unsafelyAssumingElementType: Type<Element>()]
      - rhs[unsafelyAssumingElementType: Type<Element>()]
    }
    subtract = { lhs, rhs in
      lhs[unsafelyAssumingElementType: Type<Element>()]
      -= rhs[unsafelyAssumingElementType: Type<Element>()]
    }
    /*
    scale = { 
      storage[as: storageType].scale(by: factor)
    }
    dot = { storage, others in
      storage[as: storageType].dot(.init(unsafelyDowncasting: others))
    }
    jacobians = { storage, scalar in
      storage[as: storageType].jacobians(scalar: scalar)
    }
     */
    super.init(Type<Element>())
  }
}

extension AnyArrayBuffer where Dispatch == VectorArrayDispatch {
  /// Creates an instance from a typed buffer of `Element`
  init<Element: EuclideanVectorN>(_ src: ArrayBuffer<Element>) {
    self.init(
      storage: src.storage,
      dispatch: VectorArrayDispatch(Type<Element>()))
  }
}

extension AnyArrayBuffer where Dispatch: VectorArrayDispatch {
  /// Adds `others` to `self`.
  ///
  /// - Requires: `others.count == count`.
  mutating func add(_ others: AnyElementArrayBuffer) {
    ensureUniqueStorage()
    withUnsafeMutablePointer(to: &storage) { dispatch.add($0, others) }
  }

  /// Scales each element of `self` by `factor`.
  mutating func scale(by factor: Double) {
    ensureUniqueStorage()
    withUnsafeMutablePointer(to: &storage) { dispatch.scale($0, factor) }
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
    withUnsafePointer(to: storage) { dispatch.jacobians($0, scalar) }
  }

  var scalarJacobianType: Any.Type {
    dispatch.scalarJacobianType
  }
}

extension AnyArrayBuffer: Equatable
  where Dispatch: VectorArrayDispatch
{
  public static func == (_ lhs: Self, _ rhs: Self) -> Bool {
    withUnsafePointer(to: &lhs.storage) { dispatch.equals($0, rhs) }
  }
}

extension AnyArrayBuffer: AdditiveArithmetic
  where Dispatch: VectorArrayDispatch
{
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    lhs.count == 0 ? rhs : rhs.count == 0 ? lhs
      : withUnsafePointer(to: &lhs.storage) { .init(dispatch.adding($0, rhs)) }
  }
  
  public static func - (_ lhs: Self, _ rhs: Self) -> Self {
    rhs.count == 0 ? lhs
      : withUnsafePointer(to: &lhs.storage) { .init(dispatch.subtracting($0, rhs)) }
  }
  
  public static func += (_ lhs: inout Self, _ rhs: Self) {
    if rhs.count == 0 { return }
    else if lhs.count == 0 { lhs = rhs }
    else if isKnownUniquelyReferenced(&lhs.storage) {
      withUnsafePointer(to: &lhs.storage) { dispatch.add($0, rhs) }
    }
    else {
      lhs = lhs + rhs
    }
  }
  
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
    if rhs.count == 0 { return }
    else if isKnownUniquelyReferenced(&lhs.storage) {
      withUnsafePointer(to: &lhs.storage) { dispatch.subtract($0, rhs) }
    }
    else {
      lhs = lhs - rhs
    }
  }
  public static var zero: Self { .init() }
}

extension AnyArrayBuffer: Differentiable where Dispatch: DifferentiableArrayDispatch {
  public typealias TangentVector = AnyVectorArrayBuffer
  
  /// Returns the zero `TangentVector`s of the contained elements.
  var tangentVectorZeros: TangentVector {
    withUnsafeMutableBufferPointer { vs in
      .init(vs.lazy.map { $0.zeroTangentVector })
    }
  }

  public mutating func move(along direction: TangentVector) {
    ensureUniqueStorage()
    withUnsafeMutablePointer(to: &storage) { dispatch.move($0, directions) }
  }

  public var zeroTangentVectorInitializer: () -> TangentVector {
    { mapValues { _ in .zero } }
  }
  
  /// Moves each element along the corresponding element of `directions`.
  mutating func move(along directions: AnyElementArrayBuffer) {
    ensureUniqueStorage()
    withUnsafeMutablePointer(to: &storage) { dispatch.move($0, directions) }
  }
}
*/
