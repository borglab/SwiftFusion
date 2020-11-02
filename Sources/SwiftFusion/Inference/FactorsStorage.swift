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

/// Contiguous storage for factors of statically unknown type.

import PenguinParallel
import PenguinStructures

// MARK: - Algorithms on arrays of `Factor`s.

extension ArrayStorage where Element: Factor {
  /// Returns the errors, at `x`, of the factors.
  func errors(at x: VariableAssignments) -> [Double] {
    Element.Variables.withBufferBaseAddresses(x) { varsBufs in
      Array<Double>(unsafeUninitializedCapacity: self.count) { (b, resultCount) in
        ComputeThreadPools.local.parallelFor(n: self.count) { (i, _) in
          let f = self[i]
          b[i] = f.error(at: Element.Variables(at: f.edges, in: varsBufs))
        }
        resultCount = self.count
      }
    }
  }
}

// MARK: - Algorithms on arrays of `VectorFactor`s.

extension ArrayStorage where Element: VectorFactor {
  /// Returns the error vectors, at `x`, of the factors.
  func errorVectors(at x: VariableAssignments) -> ArrayBuffer<Element.ErrorVector> {
    Element.Variables.withBufferBaseAddresses(x) { varsBufs in
      .init(
        self.lazy.map { f in
          f.errorVector(at: Element.Variables(at: f.edges, in: varsBufs)) })
    }
  }

  /// Increments `result` by the gradients of `self`'s errors at `x`.
  func accumulateErrorGradient(
    at x: VariableAssignments,
    into result: inout AllVectors
  ) {
    typealias Variables = Element.Variables
    typealias LVariables = Element.LinearizableComponent.Variables
    typealias GradVariables = LVariables.TangentVector
    Variables.withBufferBaseAddresses(x) { varsBufs in
      GradVariables.withMutableBufferBaseAddresses(&result) { gradBufs in
        for factor in self {
          let vars = Variables(at: factor.edges, in: varsBufs)
          let (lFactor, lVars) = factor.linearizableComponent(at: vars)
          let gradIndices = LVariables.linearized(lFactor.edges)
          let grads = GradVariables(at: gradIndices, in: GradVariables.withoutMutation(gradBufs))
          let newGrads = grads + gradient(at: lVars) { lFactor.errorVector(at: $0).squaredNorm }
          newGrads.assign(into: gradIndices, in: gradBufs)
        }
      }
    }
  }
}

// MARK: - Algorithms on arrays of `GaussianFactor`s.

extension ArrayStorage where Element: GaussianFactor {
  /// Returns the error vectors, at `x`, of the factors.
  func errorVectors(at x: VariableAssignments) -> ArrayBuffer<Element.ErrorVector> {
    Element.Variables.withBufferBaseAddresses(x) { varsBufs in
      .init(
        lazy.map { f in
          f.errorVector(at: Element.Variables(at: f.edges, in: varsBufs))
      })
    }
  }

  /// Returns the linear component of `errorVectors` at `x`.
  func errorVectors_linearComponent(_ x: VariableAssignments) -> ArrayBuffer<Element.ErrorVector> {
    Element.Variables.withBufferBaseAddresses(x) { varsBufs in
      // Optimized version of
      //
      // ```
      // .init(lazy.map { f in
      //   f.errorVector_linearComponent(Element.Variables(varsBufs, indices: f.edges))
      // })
      // ```
      //
      // I belive the main reason this speeds things up is that it makes it easier for the
      // optimizer to see that we call `Element.Variables(varsBufs, indices: f.edges)` multiple
      // times, which encourages it to destructure the `varsBufs` tuple once before the loop
      // instead of once per iteration.
      withUnsafeMutableBufferPointer { fs in
        .init(count: fs.count, minimumCapacity: fs.count) { baseAddress in
          for i in 0..<fs.count {
            (baseAddress + i).initialize(
              to: fs[i].errorVector_linearComponent(
                Element.Variables(at: fs[i].edges, in: varsBufs)))
          }
        }
      }
    }
  }

  /// Accumulates the adjoint (aka "transpose" or "dual") of `errorVectors` at `y` into `result`.
  ///
  /// - Requires: `result` contains elements at all of `self`'s elements' inputs.
  func errorVectors_linearComponent_adjoint(
    _ y: ArrayBuffer<Element.ErrorVector>,
    into result: inout VariableAssignments
  ) {
    typealias Variables = Element.Variables
    Variables.withMutableBufferBaseAddresses(&result) { varsBufs in
      withUnsafeMutableBufferPointer { fs in
        y.withUnsafeBufferPointer { es in
          assert(fs.count == es.count)
          for i in 0..<es.count {
            let vars = Variables(at: fs[i].edges, in: Variables.withoutMutation(varsBufs))
            let newVars = vars + fs[i].errorVector_linearComponent_adjoint(es[i])
            newVars.assign(into: fs[i].edges, in: varsBufs)
          }
        }
      }
    }
  }
}

// MARK: - Type-erased arrays of `Factor`s.

typealias AnyFactorArrayBuffer = AnyArrayBuffer<FactorArrayDispatch>

/// An `AnyArrayBuffer` dispatcher that provides algorithm implementations for `Factor`
/// elements.
public class FactorArrayDispatch {
  /// The notional `Self` type of the methods in the dispatch table
  typealias Self_ = AnyArrayBuffer<AnyObject>
  
  /// A function returning the errors, at `x`, of the factors in `storage`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage` whose `Element` has a
  ///   subclass-specific `VectorFactor` type.
  final let errors: (_ self_: Self_, _ x: VariableAssignments) -> [Double]

  /// Creates an instance for elements of type `Element`.
  init<Element: Factor>(_ e: Type<Element>) {
    errors = {
      self_, x in self_[unsafelyAssumingElementType: e].storage.errors(at: x)
    }
  }
}

extension AnyArrayBuffer where Dispatch == FactorArrayDispatch {
  /// Creates an instance from a typed buffer of `Element`
  init<Dispatch: Factor>(_ src: ArrayBuffer<Dispatch>) {
    self.init(
      storage: src.storage,
      dispatch: FactorArrayDispatch(Type<Dispatch>()))
  }
}

extension AnyArrayBuffer where Dispatch: FactorArrayDispatch {
  /// Returns the errors, at `x`, of the factors.
  func errors(at x: VariableAssignments) -> [Double] {
    dispatch.errors(self.upcast, x)
  }
}

// MARK: - Type-erased arrays of `VectorFactor`s.

typealias AnyVectorFactorArrayBuffer = AnyArrayBuffer<VectorFactorArrayDispatch>

/// An `AnyArrayBuffer` dispatcher that provides algorithm implementations for `VectorFactor`
/// elements.
public class VectorFactorArrayDispatch: FactorArrayDispatch {
  /// A function returning the error vectors, at `x`, of the factors in `storage`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage` whose `Element` has a
  ///   subclass-specific `VectorFactor` type.
  final let errorVectors:
    (_ self_: Self_, _ x: VariableAssignments) -> AnyVectorArrayBuffer

  /// A function returning the linearizations, at `x`, of the factors in `storage`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage` whose `Element` has a
  ///   subclass-specific `VectorFactor` type.
  final let linearized:
    (_ self_: Self_, _ x: VariableAssignments) -> AnyGaussianFactorArrayBuffer

  /// A function incrementing `result` by the gradients of `storage`'s factors' errors at `x`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage` whose `Element` has a
  ///   subclass-specific `VectorFactor` type.
  final let accumulateErrorGradient:
    (_ self_: Self_, _ x: VariableAssignments, _ result: inout AllVectors) -> ()

  /// Creates an instance for elements of type `Element`.
  ///
  /// The `_` argument is so that the compiler doesn't think we're trying to override the
  /// superclass init with the similar signature.
  init<Element: VectorFactor>(_ e: Type<Element>, _: () = ()) {
    errorVectors = { self_, x in
      .init(self_[unsafelyAssumingElementType: e].storage.errorVectors(at: x))
    }
    linearized = { self_, x in
      Element.linearized(self_[unsafelyAssumingElementType: e].storage, at: x)
    }
    accumulateErrorGradient = { self_, x, result in
      self_[unsafelyAssumingElementType: e].storage.accumulateErrorGradient(at: x, into: &result)
    }
    super.init(e)
  }
}

extension AnyArrayBuffer where Dispatch == VectorFactorArrayDispatch {
  /// Creates an instance from a typed buffer of `Element`
  init<Element: VectorFactor>(_ src: ArrayBuffer<Element>) {
    self.init(
      storage: src.storage,
      dispatch: VectorFactorArrayDispatch(Type<Element>()))
  }
}

extension AnyArrayBuffer where Dispatch: VectorFactorArrayDispatch {
  /// Returns the error vectors, at `x`, of the factors.
  func errorVectors(at x: VariableAssignments) -> AnyVectorArrayBuffer {
    dispatch.errorVectors(self.upcast, x)
  }

  /// Returns the linearizations, at `x`, of the factors.
  func linearized(at x: VariableAssignments) -> AnyGaussianFactorArrayBuffer {
    dispatch.linearized(self.upcast, x)
  }

  /// Increments `result` by the gradients of `self`'s errors at `x`.
  func accumulateErrorGradient(
    at x: VariableAssignments,
    into result: inout AllVectors
  ) {
    dispatch.accumulateErrorGradient(self.upcast, x, &result)
  }
}

// MARK: - Type-erased arrays of `GaussianFactor`s.

public typealias AnyGaussianFactorArrayBuffer = AnyArrayBuffer<GaussianFactorArrayDispatch>

/// An `AnyArrayBuffer` dispatcher that provides algorithm implementations for `GaussianFactor`
/// elements.
public class GaussianFactorArrayDispatch: VectorFactorArrayDispatch {
  /// A function returning the linear component of `errorVectors` at `x`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage` whose `Element` has a
  ///   subclass-specific `GaussianFactor` type.
  final let errorVectors_linearComponent:
    (_ self_: Self_,_ x: VariableAssignments) -> AnyVectorArrayBuffer

  /// A function that accumulates the adjoint (aka "transpose" or "dual") of `errorVectors` at `y`
  /// into `result`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage` whose `Element` has a
  ///   subclass-specific `GaussianFactor` type.
  /// - Requires: `y.elementType == Element.ErrorVector.self`.
  /// - Requires: `result` contains elements at all of `self`'s elements' inputs.
  final let errorVectors_linearComponent_adjoint: (
    _ self_: Self_, _ y: AnyElementArrayBuffer,
    _ result: inout VariableAssignments
  ) -> Void
  
  /// Creates an instance for elements of type `Element`.
  init<Element: GaussianFactor>(_ e: Type<Element>) 
  {
    errorVectors_linearComponent = { self_, x in
      .init(self_[unsafelyAssumingElementType: e].storage.errorVectors_linearComponent(x))
    }
    errorVectors_linearComponent_adjoint = {  self_, y, result in
      self_[unsafelyAssumingElementType: e].storage.errorVectors_linearComponent_adjoint(
        .init(unsafelyDowncasting: y), into: &result)
    }
    super.init(e)
  }
}

extension AnyArrayBuffer where Dispatch == GaussianFactorArrayDispatch {
  /// Creates an instance from a typed buffer of `Element`
  init<Element: GaussianFactor>(_ src: ArrayBuffer<Element>) {
    self.init(
      storage: src.storage,
      dispatch: GaussianFactorArrayDispatch(Type<Element>()))
  }
}

extension AnyArrayBuffer where Dispatch: GaussianFactorArrayDispatch {
  /// Returns the error vectors, at `x`, of the factors.
  func errorVectors(at x: VariableAssignments) -> AnyVectorArrayBuffer {
    dispatch.errorVectors(self.upcast, x)
  }

  /// Returns the linear component of `errorVectors` at `x`.
  func errorVectors_linearComponent(_ x: VariableAssignments) -> AnyVectorArrayBuffer {
    dispatch.errorVectors_linearComponent(self.upcast, x)
  }

  /// Accumulates the adjoint (aka "transpose" or "dual") of `errorVectors` at `y` into `result`.
  ///
  /// - Requires: `y.elementType == Element.ErrorVector.self`.
  /// - Requires: `result` contains elements at all of `self`'s elements' inputs.
  func errorVectors_linearComponent_adjoint(
    _ y: AnyElementArrayBuffer,
    into result: inout VariableAssignments
  ) {
    dispatch.errorVectors_linearComponent_adjoint(self.upcast, y, &result)
  }
}
