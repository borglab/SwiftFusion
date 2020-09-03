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

extension AnyArrayBuffer where Dispatch == VectorArrayDispatch {
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

extension AnyArrayBuffer where Dispatch == VectorArrayDispatch {
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
  @differentiable(wrt: (self, others))
  public func dot(_ others: Self) -> Double {
    dispatch.dot(self.upcast, others.upcast)
  }

  func jacobians(scalar: Double) -> AnyGaussianFactorArrayBuffer {
    dispatch.jacobians(self.upcast, scalar)
  }

  @usableFromInline
  @derivative(of: +, wrt: (lhs, rhs))
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
  static func vjp_timesEqual(lhs: Double, rhs: Self)
    -> (value: Self, pullback: (TangentVector)->(Double, TangentVector))
  {
    return (lhs * rhs, { tangent in (rhs.dot(tangent), lhs * tangent) })
  }

  @derivative(of: dot)
  @usableFromInline
  func vjp_dot(_ other: Self)
    -> (value: Double, pullback: (Double)->(TangentVector, TangentVector))
  {
    return (self.dot(other), { r in (r * other, r * self) })
  }
}

extension AnyArrayBuffer: AdditiveArithmetic where Dispatch == VectorArrayDispatch {
  public static var zero: Self { .init(ArrayBuffer<Vector1>()) }
}

extension AnyArrayBuffer: Differentiable where Dispatch: DifferentiableArrayDispatch {
  public typealias TangentVector = AnyVectorArrayBuffer

  public mutating func move(along direction: TangentVector) {
    dispatch.move(&self.upcast, direction.upcast)
  }

  public var zeroTangentVectorInitializer: () -> TangentVector {
    { .init(ArrayBuffer<Vector1>()) }
  }
}

extension AnyVectorArrayBuffer: Vector {
  
  public var scalars: AnyMutableCollection<Double> {
    get { dispatch.scalars_get(self.upcast) }
    set { dispatch.scalars_set(&self.upcast, newValue) }
  }

  public var dimension: Int {
    dispatch.dimension(self.upcast)
  }
}
