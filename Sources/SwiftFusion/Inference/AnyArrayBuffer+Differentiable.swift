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

extension AnyArrayBuffer: Differentiable where Dispatch: DifferentiableArrayDispatch {
  public typealias TangentVector = AnyVectorArrayBuffer

  public mutating func move(along direction: TangentVector) {
    dispatch.move(&self.upcast, direction.upcast)
  }

  public var zeroTangentVectorInitializer: () -> TangentVector {
    { .init(zero: ()) }
  }
}

public typealias AnyDifferentiableArrayBuffer = AnyArrayBuffer<DifferentiableArrayDispatch>

/// An `AnyArrayBuffer` dispatcher that provides algorithm implementations for `Differentiable`
/// elements.
public class DifferentiableArrayDispatch {
  /// The notional `Self` type of the methods in the dispatch table
  typealias Self_ = AnyArrayBuffer<AnyObject>
  
  /// The `TangentVector` type of the elements.
  final let tangentVectorType: Any.Type

  /// A function returning the zero `TangentVector`s for each of the elements in `self_`.
  ///
  /// - Requires: the elements of `self_` arguments have the type with which this dispatcher was
  ///   initialized.
  final let tangentVectorZeros: (_ self_: Self_) -> AnyArrayBuffer<VectorArrayDispatch>

  /// A function that moves each element of its first argument along the
  /// corresponding element of `directions`.
  ///
  /// - Requires: where `Element` is the type with which this dispatcher was initialized:
  ///   - the elements of `self_` are of type `Element`.
  ///   - the elements of directions are of type `Element.TangentVector`
  final let move: (inout Self_, _ directions: AnyElementArrayBuffer) -> Void

  /// Creates an instance for elements of type `Element`.
  init<Element: Differentiable>(_ e: Type<Element>) where Element.TangentVector: Vector {
    tangentVectorType = Element.TangentVector.self
    tangentVectorZeros = { self_ in
      .init(self_[unsafelyAssumingElementType: e].tangentVectorZeros)
    }
    move = { self_, directions in
      self_[unsafelyAssumingElementType: e].move(along: .init(unsafelyDowncasting: directions))
    }
  }

  /// Creates an instance for zero vector values.
  init(zero: Void) {
    tangentVectorType = Never.self
    tangentVectorZeros = { self_ in
      .init(zero: ())
    }
    move = { self_, directions in
      assert(directions.isEmpty, "can't move the zero vector")
    }
  }
}

extension AnyArrayBuffer where Dispatch == DifferentiableArrayDispatch {
  /// Creates an instance from a typed buffer of `Element`
  init<Element: Differentiable>(_ src: ArrayBuffer<Element>)
    where Element.TangentVector: Vector
  {
    self.init(
      storage: src.storage,
      dispatch: DifferentiableArrayDispatch(Type<Element>()))
  }
}

extension AnyArrayBuffer where Dispatch: DifferentiableArrayDispatch {
  /// The type of the contained elements' `TangentVector`.
  var tangentVectorType: Any.Type {
    dispatch.tangentVectorType
  }

  /// Returns the zero `TangentVector`s of the contained elements.
  var tangentVectorZeros: AnyArrayBuffer<VectorArrayDispatch> {
    dispatch.tangentVectorZeros(self.upcast)
  }

  /// Moves each element along the corresponding element of `directions`.
  mutating func move(along directions: AnyElementArrayBuffer) {
    dispatch.move(&self.upcast, directions)
  }
}

