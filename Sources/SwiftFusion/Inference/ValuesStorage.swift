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

extension ArrayBuffer where Element: Differentiable, Element.TangentVector: Vector {
  /// Returns the zero `TangentVector`s of the contained elements.
  var tangentVectorZeros: ArrayBuffer<Element.TangentVector> {
    withUnsafeBufferPointer { vs in
      .init(vs.lazy.map { $0.zeroTangentVector })
    }
  }

  /// Moves each element of `self` along the corresponding element of `directions`.
  ///
  /// - Requires: `directions.count == self.count`.
  mutating func move(along directions: ArrayBuffer<Element.TangentVector>) {
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

// MARK: - Algorithms on arrays of `Vector` values.

extension ArrayBuffer where Element: Vector {
  /// Adds `others` to `self`.
  ///
  /// - Requires: `others.count == count`.
  mutating func add(_ others: ArrayBuffer<Element>) {
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
  mutating func scale(by scalar: Double) {
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
    return withUnsafeBufferPointer { vs in
      others.withUnsafeBufferPointer { os in
        vs.indices.reduce(into: 0) { (result, i) in result += vs[i].dot(os[i]) }
      }
    }
  }

  /// Returns Jacobians that scale each element by `scalar`.
  func jacobians(scalar: Double) -> AnyGaussianFactorArrayBuffer {
    AnyGaussianFactorArrayBuffer(
      ArrayBuffer<ScalarJacobianFactor>(
        enumerated().lazy.map { (i, _) in
          .init(edges: Tuple1(TypedID<Element>(i)), scalar: scalar)
    }))
  }
}

// MARK: - Type-erased arrays of `Differentiable` values.

typealias AnyDifferentiableArrayBuffer = AnyArrayBuffer<DifferentiableArrayDispatch>

/// An `AnyArrayBuffer` dispatcher that provides algorithm implementations for `Differentiable`
/// elements.
class DifferentiableArrayDispatch {
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
    dispatch.tangentVectorZeros(self.withoutCapabilities)
  }

  /// Moves each element along the corresponding element of `directions`.
  mutating func move(along directions: AnyElementArrayBuffer) {
    dispatch.move(&self.withoutCapabilities, directions)
  }
}

// MARK: - Type-erased arrays of `Vector` values.

typealias AnyVectorArrayBuffer = AnyArrayBuffer<VectorArrayDispatch>

/// An `AnyArrayBuffer` dispatcher that provides algorithm implementations for `Vector`
/// elements.
class VectorArrayDispatch: DifferentiableArrayDispatch {
  /// The notional `Self` type of the methods in the dispatch table
  typealias Self_ = AnyArrayBuffer<AnyObject>
  
  /// A function that adds the elements of `addend` to the corresponding elements of `self_`
  ///
  /// - Requires: the arguments have elements of the same type with which this dispatcher was
  ///   initialized.
  final let add: (_ self_: inout Self_, _ addend: Self_) -> Void 

  /// A function that scales each element of `self_` by `scaleFactor`
  ///
  /// - Requires: `self_` has elements of the type with which this dispatcher was initialized.
  final let scale: (_ self_: inout Self_, _ scaleFactor: Double) -> Void

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

  /// Creates an instance for elements of type `Element`.
  init<Element: Vector>(_: Type<Element>, _: () = ()) {
    let e = Type<Element>()
    add = { buffer, others in
      buffer[unsafelyAssumingElementType: e].add(.init(unsafelyDowncasting: others))
    }
    scale = { buffer, factor in
      buffer[unsafelyAssumingElementType: e].scale(by: factor)
    }
    dot = { buffer, others in
      buffer[unsafelyAssumingElementType: e].dot(.init(unsafelyDowncasting: others))
    }
    jacobians = { buffer, scalar in
      buffer[unsafelyAssumingElementType: e].jacobians(scalar: scalar)
    }
    scalarJacobianType = ScalarJacobianFactor<Element>.self
    super.init(Type<Element>())
  }
}

extension AnyArrayBuffer where Dispatch == VectorArrayDispatch {
  /// Creates an instance from a typed buffer of `Element`
  init<Element: Vector>(_ src: ArrayBuffer<Element>) {
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
    dispatch.add(&self.withoutCapabilities, others)
  }

  /// Scales each element of `self` by `factor`.
  mutating func scale(by factor: Double) {
    dispatch.scale(&self.withoutCapabilities, factor)
  }

  /// Returns the dot product of `self` with `others`.
  ///
  /// This is the sum of the dot products of corresponding elements.
  ///
  /// - Requires: `others.count == count`.
  func dot(_ others: AnyElementArrayBuffer) -> Double {
    dispatch.dot(self.withoutCapabilities, others)
  }

  func jacobians(scalar: Double) -> AnyGaussianFactorArrayBuffer {
    dispatch.jacobians(self.withoutCapabilities, scalar)
  }

  var scalarJacobianType: Any.Type {
    dispatch.scalarJacobianType
  }
}
