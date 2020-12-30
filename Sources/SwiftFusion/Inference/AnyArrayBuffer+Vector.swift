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

extension AnyVectorArrayBuffer {
  /// Creates an instance from an untyped buffer with a dispatcher that is at least as specific as
  /// `Dispatch`.
  ///
  /// This initializer is effectively an up-cast.
  init<D: VectorArrayDispatch>(_ src: AnyArrayBuffer<D>) {
    self.init(unsafelyCasting: src)
  }
}

extension AnyArrayBuffer: Equatable where Dispatch: VectorArrayDispatch {
  /// Returns `true` iff `lhs` and `rhs` are equivalent tensors.
  ///
  /// - Requires: `lhs.tensorShapeIsCompatible(withThatOf: rhs)`.
  public static func ==(lhs: Self, rhs: Self) -> Bool {
    lhs.dispatch.equals(.init(lhs), .init(rhs))
  }
}

extension AnyVectorArrayBuffer {
  /// Returns the elementwise sum of `lhs` and `rhs`
  ///
  /// - Requires: the arguments have elements of the same type.
  @differentiable
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    .init(unsafelyCasting: lhs.dispatch.sum(lhs.upcast, rhs.upcast))
  }
  
  /// Returns the elementwise difference of `lhs` and `rhs`.
  ///
  /// - Requires: the arguments have elements of the same type.
  @differentiable
  public static func - (_ lhs: Self, _ rhs: Self) -> Self {
    .init(unsafelyCasting: lhs.dispatch.difference(lhs.upcast, rhs.upcast))
  }

  /// Accumulates the elements of rhs into those of lhs via addition.
  ///
  /// - Requires: the arguments have elements of the same type.
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
    lhs.dispatch.add(&lhs.upcast, rhs.upcast)
  }
  
  /// Accumulates the elements of rhs into those of lhs via subtraction.
  ///
  /// - Requires: the arguments have elements of the same type.
  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
    lhs.dispatch.subtract(&lhs.upcast, rhs.upcast)
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
    lhs.dispatch.scale(&lhs.upcast, rhs)
  }

  @differentiable
  public static func * (_ lhs: Double, _ rhs: Self) -> Self {
    .init(unsafelyCasting: rhs.dispatch.scaled(rhs.upcast, lhs))
  }

  /// Returns the dot product of `self` with `others`.
  ///
  /// This is the sum of the dot products of corresponding elements.
  ///
  /// - Requires: `others.count == count`.
  @differentiable
  public func dot(_ others: Self) -> Double {
    dispatch.dot(self.upcast, others.upcast)
  }

  func jacobians(scalar: Double) -> AnyGaussianFactorArrayBuffer {
    dispatch.jacobians(self.upcast, scalar)
  }

  @usableFromInline
  @derivative(of: +)
  static func vjp_plus(lhs: Self, rhs: Self) 
    -> (value: Self, pullback: (TangentVector)->(TangentVector, TangentVector))
  {
    (lhs + rhs, { x in (x, x) })
  }
  
  @usableFromInline
  @derivative(of: -)
  static func vjp_minus(lhs: Self, rhs: Self) 
    -> (value: Self, pullback: (TangentVector)->(TangentVector, TangentVector))
  {
    (lhs - rhs, { x in (x, x.zeroTangentVector - x) })
  }
  
  @usableFromInline
  @derivative(of: +=)
  static func vjp_plusEquals(lhs: inout Self, rhs: Self) 
    -> (value: Void, pullback: (inout TangentVector)->(TangentVector))
  {
    lhs += rhs
    return ((), { x in x })
  }
  
  @usableFromInline
  @derivative(of: -=)
  static func vjp_minusEquals(lhs: inout Self, rhs: Self) 
    -> (value: Void, pullback: (inout TangentVector)->(TangentVector))
  {
    lhs -= rhs
    return ((), { x in x.zeroTangentVector - x })
  }
  
  @derivative(of: *)
  @usableFromInline
  static func vjp_times(lhs: Double, rhs: Self)
    -> (value: Self, pullback: (TangentVector)->(Double, TangentVector))
  {
    return (lhs * rhs, { tangent in (rhs.dot(tangent), lhs * tangent) })
  }

  @derivative(of: *=)
  @usableFromInline
  static func vjp_timesEqual(lhs: inout Self, rhs: Double)
    -> (value: Void, pullback: (inout TangentVector) -> Double)
  {
    defer { lhs *= rhs }
    return ((), { [lhs = lhs] dlhs in
      let drhs = lhs.dot(dlhs)
      dlhs *= rhs
      return drhs
    })
  }

  @derivative(of: dot)
  @usableFromInline
  func vjp_dot(_ other: Self)
    -> (value: Double, pullback: (Double)->(TangentVector, TangentVector))
  {
    return (self.dot(other), { r in (r * other, r * self) })
  }
}

extension AnyVectorArrayBuffer: AdditiveArithmetic {
  public static var zero: Self { .init(zero: ()) }
}

extension AnyVectorArrayBuffer: Vector {
  public var scalars: AnyMutableCollection<Double> {
    get { dispatch.scalars_get(self.upcast) }
    set { dispatch.scalars_set(&self.upcast, newValue) }
  }

  public var dimension: Int {
    dispatch.dimension(self.upcast)
  }
  
  // DWA TODO: Remove these two things when they stop being requirements of Vector.

  /// Returns the result of calling `body` on the scalars of `self`.
  public func withUnsafeBufferPointer<R>(
    _ body: (UnsafeBufferPointer<Double>) throws -> R
  ) rethrows -> R {
    try Array(scalars).withUnsafeBufferPointer(body)
  }

  /// Returns the result of calling `body` on the scalars of `self`.
  public mutating func withUnsafeMutableBufferPointer<R>(
    _ body: (inout UnsafeMutableBufferPointer<Double>) throws -> R
  ) rethrows -> R {
    var buffer = Array(scalars)
    let r = try buffer.withUnsafeMutableBufferPointer { try body(&$0) }
    scalars.assign(buffer)
    return r
  }
}

// MARK: - Type-erased arrays of `Vector` values.

public typealias AnyVectorArrayBuffer = AnyArrayBuffer<VectorArrayDispatch>

/// An dispatcher that provides algorithm implementations for `Vector` semantics of
/// `AnyArrayBuffer`s having `Vector` elements.
public class VectorArrayDispatch: DifferentiableArrayDispatch {
  /// A function returning `true` iff `lhs` is equal to `rhs`
  ///
  /// - Requires: the arguments have elements of the same type with which this dispatcher was
  ///   initialized.
  final let equals: (_ lhs: AnyVectorArrayBuffer, _ rhs: AnyVectorArrayBuffer) -> Bool

  /// A function that returns the elementwise sum of `lhs` and `rhs`
  ///
  /// - Requires: the arguments have elements of the same type with which this dispatcher was
  ///   initialized.
  final let sum: (_ lhs: Self_, _ rhs: Self_) -> AnyVectorArrayBuffer 

  /// A function that adds the elements of `addend` to the corresponding elements of `self_`
  ///
  /// - Requires: the arguments have elements of the same type with which this dispatcher was
  ///   initialized.
  final let add: (_ self_: inout Self_, _ addend: Self_) -> Void 

  /// A function that returns the elementwise difference of `lhs -  rhs`
  ///
  /// - Requires: the arguments have elements of the same type with which this dispatcher was
  ///   initialized.
  final let difference: (_ lhs: Self_, _ rhs: Self_) -> AnyVectorArrayBuffer
  
  /// A function that subtracts the elements of `subtrahend` from the corresponding elements of
  /// `self_`
  ///
  /// - Requires: the arguments have elements of the same type with which this dispatcher was
  ///   initialized.
  final let subtract: (_ self_: inout Self_, _ subtrahend: Self_) -> Void 

  /// A function that scales each element of `self_` by `scaleFactor`
  ///
  /// - Requires: `self_` has elements of the type with which this dispatcher was initialized.
  final let scale: (_ self_: inout Self_, _ scaleFactor: Double) -> Void

  /// A function that returns the elements of `self_` scaled by `scaleFactor`
  ///
  /// - Requires: `self_` has elements of the type with which this dispatcher was initialized.
  final let scaled: (_ self_: Self_, _ scaleFactor: Double) -> AnyVectorArrayBuffer

  /// A function that returns the sum of the dot products of corresponding elements of `lhs` and
  /// `rhs`.
  ///
  /// - Requires: the arguments have elements of the same type with which this dispatcher was
  ///   initialized.
  final let dot: (_ lhs: Self_, _ rhs: Self_) -> Double

  /// A function that maps each element of `self_` into a Jacobian factor that scales the element by
  /// `scalar`.
  ///
  /// - Requires: `self_` has elements of the type with which this dispatcher was initialized.
  final let jacobians: (_ self_: Self_, _ scalar: Double) -> AnyGaussianFactorArrayBuffer

  /// The type of a `ScalarJacobianFactor` that scales the elements.
  final let scalarJacobianType: Any.Type

  final let scalars_get: (_ self_: Self_) -> AnyMutableCollection<Double>
  final let scalars_set: (_ self_: inout Self_, _ newValue: AnyMutableCollection<Double>) -> Void

  final let dimension: (_ self_: Self_) -> Int
  
  /// Creates an instance for elements of type `Element`.
  init<Element: Vector>(_ e: Type<Element>, _: () = ()) {
    equals = { lhs, rhs in
      lhs[unsafelyAssumingElementType: e] == rhs[unsafelyAssumingElementType: e]
    }
    sum = { lhs, rhs in
      .init(lhs[unsafelyAssumingElementType: e] + rhs[unsafelyAssumingElementType: e])
    }
    add = { lhs, rhs in
      lhs[unsafelyAssumingElementType: e] += rhs[unsafelyAssumingElementType: e]
    }
    difference = { lhs, rhs in
      .init(
        lhs[unsafelyAssumingElementType: e] - rhs[unsafelyAssumingElementType: e])
    }
    subtract = { lhs, rhs in
      lhs[unsafelyAssumingElementType: e] -= rhs[unsafelyAssumingElementType: e]
    }
    scale = { target, scaleFactor in
      target[unsafelyAssumingElementType: e] *= scaleFactor
    }
    scaled = { target, scaleFactor in
      .init(scaleFactor * target[unsafelyAssumingElementType: e])
    }
    dot = { lhs, rhs in
      lhs[unsafelyAssumingElementType: e].dot(rhs[unsafelyAssumingElementType: e])
    }
    jacobians = { target, scaleFactor in
      .init(target[unsafelyAssumingElementType: e].jacobians(scalar: scaleFactor))
    }
    scalarJacobianType = ScalarJacobianFactor<Element>.self

    scalars_get = { self_ in .init(FlattenedScalars(self_[unsafelyAssumingElementType: e])) }
    scalars_set = { self_, newValue in
      // TODO: see if we need to and can make this more efficient.
      self_[unsafelyAssumingElementType: e].scalars.assign(newValue)
    }

    dimension = { self_ in
      self_[unsafelyAssumingElementType: e].lazy.map(\.dimension).reduce(0, +)
    }
    super.init(e)
  }

  /// Creates an instance for zero vector values.
  override init(zero _: ()) {
    equals = { lhs, rhs in
      rhs.isEmpty || rhs == lhs
    }
    sum = { lhs, rhs in .init(unsafelyCasting: rhs) }
    add = { lhs, rhs in
      if !rhs.isEmpty {
        lhs = rhs
      }
    }
    difference = { lhs, rhs in
      rhs.isEmpty ? .init(unsafelyCasting: lhs) : -1.0 * AnyVectorArrayBuffer(unsafelyCasting: rhs)
    }
    subtract = { lhs, rhs in
      if !rhs.isEmpty {
        lhs = .init(-1.0 * AnyVectorArrayBuffer(unsafelyCasting: rhs))
      }
    }
    scale = { target, scaleFactor in }
    scaled = { target, scaleFactor in
      .init(unsafelyCasting: target)
    }
    dot = { lhs, rhs in
      0.0
    }
    jacobians = { target, scaleFactor in
      fatalError("cannot implement")
    }
    scalarJacobianType = ScalarJacobianFactor<Vector1>.self

    scalars_get = { self_ in
      .init(FlattenedScalars(self_[unsafelyAssumingElementType: Type<Vector1>()]))
    }
    scalars_set = { self_, newValue in
      assert(newValue.isEmpty || newValue.allSatisfy { $0 == 0 })
    }

    dimension = { self_ in
      0 // fatalError()?
    }
    super.init(zero: ())
  }
}

extension AnyVectorArrayBuffer {
  /// Creates an instance from a typed buffer of `Element`
  public init<Element: Vector>(_ src: ArrayBuffer<Element>) {
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
    dispatch.add(&self.upcast, others)
  }

  /// Scales each element of `self` by `factor`.
  mutating func scale(by factor: Double) {
    dispatch.scale(&self.upcast, factor)
  }

  var scalarJacobianType: Any.Type {
    dispatch.scalarJacobianType
  }
}
