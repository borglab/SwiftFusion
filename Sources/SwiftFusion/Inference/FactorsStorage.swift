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

import PenguinStructures

// MARK: - Algorithms on arrays of `Factor`s.

extension ArrayStorage where Element: Factor {
  /// Returns the errors, at `x`, of the factors.
  func errors(at x: VariableAssignments) -> [Double] {
    Element.Variables.withBufferBaseAddresses(x) { varsBufs in
      self.map { f in
        f.error(at: Element.Variables(at: f.edges, in: varsBufs))
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

  /// Returns the linearized factors at `x`.
  func linearized<Linearization: LinearApproximationFactor>(at x: VariableAssignments)
    -> ArrayBuffer<Linearization>
  where Linearization.Variables == Element.LinearizableComponent.Variables.TangentVector,
        Linearization.ErrorVector == Element.ErrorVector
  {
    Element.Variables.withBufferBaseAddresses(x) { varsBufs in
      .init(
        self.lazy.map { f in
          let (fLinearizable, xLinearizable) =
            f.linearizableComponent(at: Element.Variables(at: f.edges, in: varsBufs))
        return Linearization(linearizing: fLinearizable, at: xLinearizable)
        })
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
class FactorArrayDispatch {
  /// A function returning the errors, at `x`, of the factors in `storage`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage` whose `Element` has a
  ///   subclass-specific `VectorFactor` type.
  final let errors: (_ storage: UnsafeRawPointer, _ x: VariableAssignments) -> [Double]

  /// Creates an instance for elements of type `Element`.
  init<Element: Factor>(_: Type<Element>) {
    let storageType = Type<ArrayStorage<Element>>()
    errors = { storage, x in storage[as: storageType].errors(at: x) }
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
    withUnsafePointer(to: storage) { dispatch.errors($0, x) }
  }
}

// MARK: - Type-erased arrays of `VectorFactor`s.

typealias AnyVectorFactorArrayBuffer = AnyArrayBuffer<VectorFactorArrayDispatch>

/// An `AnyArrayBuffer` dispatcher that provides algorithm implementations for `VectorFactor`
/// elements.
class VectorFactorArrayDispatch: FactorArrayDispatch {
  /// A function returning the error vectors, at `x`, of the factors in `storage`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage` whose `Element` has a
  ///   subclass-specific `VectorFactor` type.
  final let errorVectors:
    (_ storage: UnsafeRawPointer, _ x: VariableAssignments) -> AnyVectorArrayBuffer

  /// A function returning the linearizations, at `x`, of the factors in `storage`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage` whose `Element` has a
  ///   subclass-specific `VectorFactor` type.
  final let linearized:
    (_ storage: UnsafeRawPointer, _ x: VariableAssignments) -> AnyGaussianFactorArrayBuffer

  /// Creates an instance for elements of type `Element` using the given `Linearization`.
  init<Element: VectorFactor, Linearization: LinearApproximationFactor>(
    _: Type<Element>, linearization: Type<Linearization>
  )
    where Linearization.Variables == Element.LinearizableComponent.Variables.TangentVector,
          Linearization.ErrorVector == Element.ErrorVector
  {
    let storageType = Type<ArrayStorage<Element>>()

    errorVectors = { storage, x in
      .init(storage[as: storageType].errorVectors(at: x))
    }
    
    linearized = { storage, x in
      .init(storage[as: storageType].linearized(at: x) as ArrayBuffer<Linearization>)
    }
    super.init(Type<Element>())
  }
}

extension AnyArrayBuffer where Dispatch == VectorFactorArrayDispatch {
  /// Creates an instance from a typed buffer of `Element`
  init<Element: VectorFactor>(_ src: ArrayBuffer<Element>) {
    let dispatch: VectorFactorArrayDispatch
    let elementType = Type<Element>()
    typealias TangentVector = Element.LinearizableComponent.Variables.TangentVector
    typealias Linearization<A> = Type<JacobianFactor<A, Element.ErrorVector>>
      where A: SourceInitializableCollection, A.Element: Vector & DifferentiableVariableTuple

    // For small dimensions, we use a fixed size array in the linearization because the allocation
    // and indirection overheads of a dynamically sized array are releatively big.
    //
    // For larger dimensions, dynamically sized array are more convenient (no need to define a
    // new type and case for each size) and in some cases also happen to be faster than fixed size
    // arrays.
    //
    // We chose 4 as the cutoff based on benchmark results:
    // - "Pose2SLAM.FactorGraph", which heavily uses 3 dimensional error vectors, is ~30% faster
    //   with a fixed size array than with a dynamically sized array.
    // - "Pose3SLAM.FactorGraph", which heavily uses 6 dimensional error vectors, is ~3% faster
    //   with a dynamically sized array than with a fixed size array.
    // - "Pose3SLAM.sphere2500", which heavily uses 12 dimensional error vectors, has the same
    //   performance with fixed size and dynamically sized arrays.
    switch Element.ErrorVector.dimension {
    case 1:
      dispatch = .init(elementType, linearization: Linearization<Array1<TangentVector>>())
    case 2:
      dispatch = .init(elementType, linearization: Linearization<Array2<TangentVector>>())
    case 3:
      dispatch = .init(elementType, linearization: Linearization<Array3<TangentVector>>())
    case 4:
      dispatch = .init(elementType, linearization: Linearization<Array4<TangentVector>>())
    default:
      dispatch = .init(elementType, linearization: Linearization<Array<TangentVector>>())
    }

    self.init(storage: src.storage, dispatch: dispatch)
  }
}

extension AnyArrayBuffer where Dispatch: VectorFactorArrayDispatch {
  /// Returns the error vectors, at `x`, of the factors.
  func errorVectors(at x: VariableAssignments) -> AnyVectorArrayBuffer {
    withUnsafePointer(to: storage) { dispatch.errorVectors($0, x) }
  }

  /// Returns the linearizations, at `x`, of the factors.
  func linearized(at x: VariableAssignments) -> AnyGaussianFactorArrayBuffer {
    withUnsafePointer(to: storage) { dispatch.linearized($0, x) }
  }
}

// MARK: - Type-erased arrays of `GaussianFactor`s.

typealias AnyGaussianFactorArrayBuffer = AnyArrayBuffer<GaussianFactorArrayDispatch>

/// An `AnyArrayBuffer` dispatcher that provides algorithm implementations for `GaussianFactor`
/// elements.
class GaussianFactorArrayDispatch: VectorFactorArrayDispatch {
  /// A function returning the linear component of `errorVectors` at `x`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage` whose `Element` has a
  ///   subclass-specific `GaussianFactor` type.
  final let errorVectors_linearComponent:
    (_ storage: UnsafeRawPointer,_ x: VariableAssignments) -> AnyVectorArrayBuffer

  /// A function that accumulates the adjoint (aka "transpose" or "dual") of `errorVectors` at `y`
  /// into `result`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage` whose `Element` has a
  ///   subclass-specific `GaussianFactor` type.
  /// - Requires: `y.elementType == Element.ErrorVector.self`.
  /// - Requires: `result` contains elements at all of `self`'s elements' inputs.
  final let errorVectors_linearComponent_adjoint: (
    _ storage: UnsafeRawPointer, _ y: AnyElementArrayBuffer,
    _ result: inout VariableAssignments
  ) -> Void
  
  /// Creates an instance for elements of type `Element`.
  init<Element: GaussianFactor>(_: Type<Element>) 
  {
    let storageType = Type<ArrayStorage<Element>>()
    errorVectors_linearComponent = { storage, x in
      .init(storage[as: storageType].errorVectors_linearComponent(x))
    }
    errorVectors_linearComponent_adjoint = {  storage, y, result in
      storage[as: storageType].errorVectors_linearComponent_adjoint(
        .init(unsafelyDowncasting: y), into: &result)
    }
    super.init(Type<Element>(), linearization: Type<IdentityLinearizationFactor<Element>>())
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
    withUnsafePointer(to: storage) { dispatch.errorVectors($0, x) }
  }

  /// Returns the linear component of `errorVectors` at `x`.
  func errorVectors_linearComponent(_ x: VariableAssignments) -> AnyVectorArrayBuffer {
    withUnsafePointer(to: storage) { dispatch.errorVectors_linearComponent($0, x) }
  }

  /// Accumulates the adjoint (aka "transpose" or "dual") of `errorVectors` at `y` into `result`.
  ///
  /// - Requires: `y.elementType == Element.ErrorVector.self`.
  /// - Requires: `result` contains elements at all of `self`'s elements' inputs.
  func errorVectors_linearComponent_adjoint(
    _ y: AnyElementArrayBuffer,
    into result: inout VariableAssignments
  ) {
    withUnsafePointer(to: storage) {
      dispatch.errorVectors_linearComponent_adjoint($0, y, &result)
    }
  }
}
