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

extension AnyArrayBuffer where Dispatch: VectorArrayDispatch
{
  /// Returns the elementwise sum of `lhs` and `rhs`
  ///
  /// - Requires: the arguments have elements of the same type.
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    .init(unsafelyCasting: lhs.dispatch.sum(lhs.upcast, rhs.upcast))
  }
  
  /// Returns the elementwise difference of `lhs` and `rhs`.
  ///
  /// - Requires: the arguments have elements of the same type.
  public static func - (_ lhs: Self, _ rhs: Self) -> Self {
    .init(unsafelyCasting: lhs.dispatch.difference(lhs.upcast, rhs.upcast))
  }

  /// Accumulates the elements of rhs into those of lhs via addition.
  ///
  /// - Requires: the arguments have elements of the same type.
  public static func += (_ lhs: inout Self, _ rhs: Self) {
    lhs.dispatch.add(&lhs.upcast, rhs.upcast)
  }
  
  /// Accumulates the elements of rhs into those of lhs via subtraction.
  ///
  /// - Requires: the arguments have elements of the same type.
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
    lhs.dispatch.subtract(&lhs.upcast, rhs.upcast)
  }
}

extension AnyArrayBuffer: AdditiveArithmetic where Dispatch == VectorArrayDispatch
{
  public static var zero: Self { .init(ArrayBuffer<Vector1>()) }
}

extension AnyArrayBuffer: Differentiable where Dispatch: DifferentiableArrayDispatch {
  public typealias TangentVector = AnyVectorArrayBuffer

  public mutating func move(along direction: TangentVector) {
    dispatch.move(&self.upcast, direction.upcast)
  }

  public var zeroTangentVectorInitializer: () -> TangentVector {
    { .zero }
  }
}
