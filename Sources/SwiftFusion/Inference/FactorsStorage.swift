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
    Element.Variables.withVariableBufferBaseUnsafePointers(x) { varsBufs in
      map { f in
        f.error(at: Element.Variables(varsBufs, indices: f.edges))
      }
    }
  }
}

// MARK: - Algorithms on arrays of `VectorFactor`s.

extension ArrayStorage where Element: VectorFactor {
  /// Returns the error vectors, at `x`, of the factors.
  func errorVectors(at x: VariableAssignments) -> ArrayBuffer<Element.ErrorVector> {
    Element.Variables.withVariableBufferBaseUnsafePointers(x) { varsBufs in
      .init(lazy.map { f in
        f.errorVector(at: Element.Variables(varsBufs, indices: f.edges))
      })
    }
  }

  /// Returns the linearized factors at `x`.
  func linearized<Linearization: LinearApproximationFactor>(at x: VariableAssignments)
    -> ArrayBuffer<Linearization>
  where Linearization.Variables == Element.LinearizableComponent.Variables.TangentVector,
        Linearization.ErrorVector == Element.ErrorVector
  {
    Element.Variables.withVariableBufferBaseUnsafePointers(x) { varsBufs in
      .init(lazy.map { f in
        let (fLinearizable, xLinearizable) =
          f.linearizableComponent(at: Element.Variables(varsBufs, indices: f.edges))
        return Linearization(linearizing: fLinearizable, at: xLinearizable)
      })
    }
  }
}

// MARK: - Algorithms on arrays of `GaussianFactor`s.

extension ArrayStorage where Element: GaussianFactor {
  /// Returns the error vectors, at `x`, of the factors.
  func errorVectors(at x: VariableAssignments) -> ArrayBuffer<Element.ErrorVector> {
    Element.Variables.withVariableBufferBaseUnsafePointers(x) { varsBufs in
      .init(lazy.map { f in
        f.errorVector(at: Element.Variables(varsBufs, indices: f.edges))
      })
    }
  }

  /// Returns the linear component of `errorVectors` at `x`.
  func errorVectors_linearComponent(_ x: VariableAssignments) -> ArrayBuffer<Element.ErrorVector> {
    Element.Variables.withVariableBufferBaseUnsafePointers(x) { varsBufs in
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
                Element.Variables(varsBufs, indices: fs[i].edges)))
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
    Element.Variables.withVariableBufferBaseUnsafeMutablePointers(&result) { varsBufs in
      withUnsafeMutableBufferPointer { fs in
        y.withUnsafeBufferPointer { es in
          for i in 0..<es.count {
            let vars = Element.Variables(varsBufs, indices: fs[i].edges)
            let newVars = vars + fs[i].errorVector_linearComponent_adjoint(es[i])
            newVars.store(into: varsBufs, indices: fs[i].edges)
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
class FactorArrayDispatch: AnyElementDispatch {
  /// Returns the errors, at `x`, of the factors in `storage`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage` whose `Element` has a
  ///   subclass-specific `VectorFactor` type.
  class func errors(_ storage: UnsafeRawPointer, at x: VariableAssignments) -> [Double] {
    fatalError("implement as in FactorArrayDispatch_")
  }
}

/// An `AnyArrayBuffer` dispatcher that provides algorithm implementations for a
/// specific `Factor` element type.
class FactorArrayDispatch_<Element: Factor>
  : FactorArrayDispatch, AnyArrayDispatch
{
  /// Returns the errors, at `x`, of the factors in `storage`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage<Element>`.
  override class func errors(_ storage: UnsafeRawPointer, at x: VariableAssignments) -> [Double] {
    asStorage(storage).errors(at: x)
  }
}

extension AnyArrayBuffer where Dispatch == FactorArrayDispatch {
  /// Creates an instance from a typed buffer of `Element`
  init<Dispatch: Factor>(_ src: ArrayBuffer<Dispatch>) {
    self.init(
      storage: src.storage,
      dispatch: FactorArrayDispatch_<Dispatch>.self)
  }
}

extension AnyArrayBuffer where Dispatch: FactorArrayDispatch {
  /// Returns the errors, at `x`, of the factors.
  func errors(at x: VariableAssignments) -> [Double] {
    withUnsafePointer(to: storage) { dispatch.errors($0, at: x) }
  }
}

// MARK: - Type-erased arrays of `VectorFactor`s.

typealias AnyVectorFactorArrayBuffer = AnyArrayBuffer<VectorFactorArrayDispatch>

/// An `AnyArrayBuffer` dispatcher that provides algorithm implementations for `VectorFactor`
/// elements.
class VectorFactorArrayDispatch: FactorArrayDispatch {
  /// Returns the error vectors, at `x`, of the factors in `storage`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage` whose `Element` has a
  ///   subclass-specific `VectorFactor` type.
  class func errorVectors(_ storage: UnsafeRawPointer, at x: VariableAssignments)
    -> AnyVectorArrayBuffer
  {
    fatalError("implement as in VectorFactorArrayDispatch_")
  }

  /// Returns the linearizations, at `x`, of the factors in `storage`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage` whose `Element` has a
  ///   subclass-specific `VectorFactor` type.
  class func linearized(_ storage: UnsafeRawPointer, at x: VariableAssignments)
    -> AnyGaussianFactorArrayBuffer
  {
    fatalError("implement as in VectorFactorArrayDispatch_")
  }
}

/// An `AnyArrayBuffer` dispatcher that provides algorithm implementations for a
/// specific `VectorFactor` element type.
class VectorFactorArrayDispatch_<
  Element: VectorFactor,
  Linearization: LinearApproximationFactor
>: VectorFactorArrayDispatch, AnyArrayDispatch
where Linearization.Variables == Element.LinearizableComponent.Variables.TangentVector,
      Linearization.ErrorVector == Element.ErrorVector
{
  /// Returns the errors, at `x`, of the factors in `storage`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage<Element>`.
  @_specialize(where Element == BetweenFactor<Pose2>, Linearization == JacobianFactor3x3_2)
  @_specialize(where Element == BetweenFactor<Pose3>, Linearization == JacobianFactor6x6_2)
  @_specialize(where Element == BetweenFactorAlternative, Linearization == JacobianFactor12x6_2)
  @_specialize(where Element == PriorFactor<Pose2>, Linearization == JacobianFactor3x3_1)
  @_specialize(where Element == PriorFactor<Pose3>, Linearization == JacobianFactor6x6_1)
  override class func errors(_ storage: UnsafeRawPointer, at x: VariableAssignments) -> [Double] {
    asStorage(storage).errors(at: x)
  }

  /// Returns the error vectors, at `x`, of the factors in `storage`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage<Element>`.
  @_specialize(where Element == BetweenFactor<Pose2>, Linearization == JacobianFactor3x3_2)
  @_specialize(where Element == BetweenFactor<Pose3>, Linearization == JacobianFactor6x6_2)
  @_specialize(where Element == BetweenFactorAlternative, Linearization == JacobianFactor12x6_2)
  @_specialize(where Element == PriorFactor<Pose2>, Linearization == JacobianFactor3x3_1)
  @_specialize(where Element == PriorFactor<Pose3>, Linearization == JacobianFactor6x6_1)
  override class func errorVectors(_ storage: UnsafeRawPointer, at x: VariableAssignments)
    -> AnyVectorArrayBuffer
  {
    .init(asStorage(storage).errorVectors(at: x))
  }

  /// Returns the linearizations, at `x`, of the factors in `storage`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage<Element>`.
  @_specialize(where Element == BetweenFactor<Pose2>, Linearization == JacobianFactor3x3_2)
  @_specialize(where Element == BetweenFactor<Pose3>, Linearization == JacobianFactor6x6_2)
  @_specialize(where Element == BetweenFactorAlternative, Linearization == JacobianFactor12x6_2)
  @_specialize(where Element == PriorFactor<Pose2>, Linearization == JacobianFactor3x3_1)
  @_specialize(where Element == PriorFactor<Pose3>, Linearization == JacobianFactor6x6_1)
  override class func linearized(_ storage: UnsafeRawPointer, at x: VariableAssignments)
    -> AnyGaussianFactorArrayBuffer
  {
    .init(asStorage(storage).linearized(at: x) as ArrayBuffer<Linearization>)
  }
}

extension AnyArrayBuffer where Dispatch == VectorFactorArrayDispatch {
  /// Creates an instance from a typed buffer of `Element`
  init<Element: VectorFactor>(_ src: ArrayBuffer<Element>) {
    let dispatch: VectorFactorArrayDispatch.Type
    switch Element.ErrorVector.dimension {
    case 1:
      dispatch = VectorFactorArrayDispatch_<
        Element,
        JacobianFactor<
          Array1<Element.LinearizableComponent.Variables.TangentVector>,
          Element.ErrorVector
        >
      >.self
    case 2:
      dispatch = VectorFactorArrayDispatch_<
        Element,
        JacobianFactor<
          Array2<Element.LinearizableComponent.Variables.TangentVector>,
          Element.ErrorVector
        >
      >.self
    case 3:
      dispatch = VectorFactorArrayDispatch_<
        Element,
        JacobianFactor<
          Array3<Element.LinearizableComponent.Variables.TangentVector>,
          Element.ErrorVector
        >
      >.self
    case 4:
      dispatch = VectorFactorArrayDispatch_<
        Element,
        JacobianFactor<
          Array4<Element.LinearizableComponent.Variables.TangentVector>,
          Element.ErrorVector
        >
      >.self
    case 5:
      dispatch = VectorFactorArrayDispatch_<
        Element,
        JacobianFactor<
          Array5<Element.LinearizableComponent.Variables.TangentVector>,
          Element.ErrorVector
        >
      >.self
    case 6:
      dispatch = VectorFactorArrayDispatch_<
        Element,
        JacobianFactor<
          Array6<Element.LinearizableComponent.Variables.TangentVector>,
          Element.ErrorVector
        >
      >.self
    case 7:
      dispatch = VectorFactorArrayDispatch_<
        Element,
        JacobianFactor<
          Array7<Element.LinearizableComponent.Variables.TangentVector>,
          Element.ErrorVector
        >
      >.self
    case 8:
      dispatch = VectorFactorArrayDispatch_<
        Element,
        JacobianFactor<
          Array8<Element.LinearizableComponent.Variables.TangentVector>,
          Element.ErrorVector
        >
      >.self
    case 9:
      dispatch = VectorFactorArrayDispatch_<
        Element,
        JacobianFactor<
          Array9<Element.LinearizableComponent.Variables.TangentVector>,
          Element.ErrorVector
        >
      >.self
    case 10:
      dispatch = VectorFactorArrayDispatch_<
        Element,
        JacobianFactor<
          Array10<Element.LinearizableComponent.Variables.TangentVector>,
          Element.ErrorVector
        >
      >.self
    case 11:
      dispatch = VectorFactorArrayDispatch_<
        Element,
        JacobianFactor<
          Array11<Element.LinearizableComponent.Variables.TangentVector>,
          Element.ErrorVector
        >
      >.self
    case 12:
      dispatch = VectorFactorArrayDispatch_<
        Element,
        JacobianFactor<
          Array12<Element.LinearizableComponent.Variables.TangentVector>,
          Element.ErrorVector
        >
      >.self
    default:
      fatalError("ErrorVector dimension \(Element.ErrorVector.dimension) not implemented")
    }

    self.init(storage: src.storage, dispatch: dispatch)
  }
}

extension AnyArrayBuffer where Dispatch: VectorFactorArrayDispatch {
  /// Returns the error vectors, at `x`, of the factors.
  func errorVectors(at x: VariableAssignments) -> AnyVectorArrayBuffer {
    withUnsafePointer(to: storage) { dispatch.errorVectors($0, at: x) }
  }

  /// Returns the linearizations, at `x`, of the factors.
  func linearized(at x: VariableAssignments) -> AnyGaussianFactorArrayBuffer {
    withUnsafePointer(to: storage) { dispatch.linearized($0, at: x) }
  }
}

// MARK: - Type-erased arrays of `GaussianFactor`s.

typealias AnyGaussianFactorArrayBuffer = AnyArrayBuffer<GaussianFactorArrayDispatch>

/// An `AnyArrayBuffer` dispatcher that provides algorithm implementations for `GaussianFactor`
/// elements.
class GaussianFactorArrayDispatch: VectorFactorArrayDispatch {
  /// Returns the linear component of `errorVectors` at `x`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage` whose `Element` has a
  ///   subclass-specific `GaussianFactor` type.
  class func errorVectors_linearComponent(
    _ storage: UnsafeRawPointer,
    _ x: VariableAssignments
  ) -> AnyVectorArrayBuffer {
    fatalError("implement as in GaussianFactorArrayDispatch_")
  }

  /// Accumulates the adjoint (aka "transpose" or "dual") of `errorVectors` at `y` into `result`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage` whose `Element` has a
  ///   subclass-specific `GaussianFactor` type.
  /// - Requires: `y.elementType == Element.ErrorVector.self`.
  /// - Requires: `result` contains elements at all of `self`'s elements' inputs.
  class func errorVectors_linearComponent_adjoint(
    _ storage: UnsafeRawPointer,
    _ y: AnyElementArrayBuffer,
    into result: inout VariableAssignments
  ) {
    fatalError("implement as in GaussianFactorArrayDispatch_")
  }
}

/// An `AnyArrayBuffer` dispatcher that provides algorithm implementations for a
/// specific `GaussianFactor` element type.
class GaussianFactorArrayDispatch_<Element: GaussianFactor>
  : GaussianFactorArrayDispatch, AnyArrayDispatch
{
  /// Returns the errors, at `x`, of the factors in `storage`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage<Element>`.
  @_specialize(where Element == JacobianFactor3x3_1)
  @_specialize(where Element == JacobianFactor3x3_2)
  @_specialize(where Element == JacobianFactor6x6_1)
  @_specialize(where Element == JacobianFactor6x6_2)
  @_specialize(where Element == JacobianFactor9x3x3_1)
  @_specialize(where Element == JacobianFactor9x3x3_2)
  @_specialize(where Element == JacobianFactor12x6_1)
  @_specialize(where Element == JacobianFactor12x6_2)
  @_specialize(where Element == ScalarJacobianFactor<Vector1>)
  @_specialize(where Element == ScalarJacobianFactor<Vector2>)
  @_specialize(where Element == ScalarJacobianFactor<Vector3>)
  @_specialize(where Element == ScalarJacobianFactor<Vector4>)
  @_specialize(where Element == ScalarJacobianFactor<Vector5>)
  @_specialize(where Element == ScalarJacobianFactor<Vector6>)
  @_specialize(where Element == ScalarJacobianFactor<Vector6>)
  @_specialize(where Element == ScalarJacobianFactor<Vector7>)
  @_specialize(where Element == ScalarJacobianFactor<Vector8>)
  @_specialize(where Element == ScalarJacobianFactor<Vector9>)
  @_specialize(where Element == ScalarJacobianFactor<Vector10>)
  @_specialize(where Element == ScalarJacobianFactor<Vector11>)
  @_specialize(where Element == ScalarJacobianFactor<Vector12>)
  override class func errors(_ storage: UnsafeRawPointer, at x: VariableAssignments) -> [Double] {
    asStorage(storage).errors(at: x)
  }

  /// Returns the error vectors, at `x`, of the factors in `storage`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage<Element>`.
  @_specialize(where Element == JacobianFactor3x3_1)
  @_specialize(where Element == JacobianFactor3x3_2)
  @_specialize(where Element == JacobianFactor6x6_1)
  @_specialize(where Element == JacobianFactor6x6_2)
  @_specialize(where Element == JacobianFactor9x3x3_1)
  @_specialize(where Element == JacobianFactor9x3x3_2)
  @_specialize(where Element == JacobianFactor12x6_1)
  @_specialize(where Element == JacobianFactor12x6_2)
  @_specialize(where Element == ScalarJacobianFactor<Vector1>)
  @_specialize(where Element == ScalarJacobianFactor<Vector2>)
  @_specialize(where Element == ScalarJacobianFactor<Vector3>)
  @_specialize(where Element == ScalarJacobianFactor<Vector4>)
  @_specialize(where Element == ScalarJacobianFactor<Vector5>)
  @_specialize(where Element == ScalarJacobianFactor<Vector6>)
  @_specialize(where Element == ScalarJacobianFactor<Vector6>)
  @_specialize(where Element == ScalarJacobianFactor<Vector7>)
  @_specialize(where Element == ScalarJacobianFactor<Vector8>)
  @_specialize(where Element == ScalarJacobianFactor<Vector9>)
  @_specialize(where Element == ScalarJacobianFactor<Vector10>)
  @_specialize(where Element == ScalarJacobianFactor<Vector11>)
  @_specialize(where Element == ScalarJacobianFactor<Vector12>)
  override class func errorVectors(_ storage: UnsafeRawPointer, at x: VariableAssignments)
    -> AnyVectorArrayBuffer
  {
    .init(asStorage(storage).errorVectors(at: x))
  }

  /// Returns the linearizations, at `x`, of the factors in `storage`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage<Element>`.
  @_specialize(where Element == JacobianFactor3x3_1)
  @_specialize(where Element == JacobianFactor3x3_2)
  @_specialize(where Element == JacobianFactor6x6_1)
  @_specialize(where Element == JacobianFactor6x6_2)
  @_specialize(where Element == JacobianFactor9x3x3_1)
  @_specialize(where Element == JacobianFactor9x3x3_2)
  @_specialize(where Element == JacobianFactor12x6_1)
  @_specialize(where Element == JacobianFactor12x6_2)
  @_specialize(where Element == ScalarJacobianFactor<Vector1>)
  @_specialize(where Element == ScalarJacobianFactor<Vector2>)
  @_specialize(where Element == ScalarJacobianFactor<Vector3>)
  @_specialize(where Element == ScalarJacobianFactor<Vector4>)
  @_specialize(where Element == ScalarJacobianFactor<Vector5>)
  @_specialize(where Element == ScalarJacobianFactor<Vector6>)
  @_specialize(where Element == ScalarJacobianFactor<Vector6>)
  @_specialize(where Element == ScalarJacobianFactor<Vector7>)
  @_specialize(where Element == ScalarJacobianFactor<Vector8>)
  @_specialize(where Element == ScalarJacobianFactor<Vector9>)
  @_specialize(where Element == ScalarJacobianFactor<Vector10>)
  @_specialize(where Element == ScalarJacobianFactor<Vector11>)
  @_specialize(where Element == ScalarJacobianFactor<Vector12>)
  override class func linearized(_ storage: UnsafeRawPointer, at x: VariableAssignments)
    -> AnyGaussianFactorArrayBuffer
  {
    // Gaussian factors are linearizations of themselves, so we can just return the factors
    // unmodified.
    .init(ArrayBuffer(asStorage(storage)))
  }

  /// Returns the linear component of `errorVectors` at `x`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage` whose `Element` has a
  ///   subclass-specific `GaussianFactor` type.
  @_specialize(where Element == JacobianFactor3x3_1)
  @_specialize(where Element == JacobianFactor3x3_2)
  @_specialize(where Element == JacobianFactor6x6_1)
  @_specialize(where Element == JacobianFactor6x6_2)
  @_specialize(where Element == JacobianFactor9x3x3_1)
  @_specialize(where Element == JacobianFactor9x3x3_2)
  @_specialize(where Element == JacobianFactor12x6_1)
  @_specialize(where Element == JacobianFactor12x6_2)
  @_specialize(where Element == ScalarJacobianFactor<Vector1>)
  @_specialize(where Element == ScalarJacobianFactor<Vector2>)
  @_specialize(where Element == ScalarJacobianFactor<Vector3>)
  @_specialize(where Element == ScalarJacobianFactor<Vector4>)
  @_specialize(where Element == ScalarJacobianFactor<Vector5>)
  @_specialize(where Element == ScalarJacobianFactor<Vector6>)
  @_specialize(where Element == ScalarJacobianFactor<Vector6>)
  @_specialize(where Element == ScalarJacobianFactor<Vector7>)
  @_specialize(where Element == ScalarJacobianFactor<Vector8>)
  @_specialize(where Element == ScalarJacobianFactor<Vector9>)
  @_specialize(where Element == ScalarJacobianFactor<Vector10>)
  @_specialize(where Element == ScalarJacobianFactor<Vector11>)
  @_specialize(where Element == ScalarJacobianFactor<Vector12>)
  override class func errorVectors_linearComponent(
    _ storage: UnsafeRawPointer,
    _ x: VariableAssignments
  ) -> AnyVectorArrayBuffer {
    .init(asStorage(storage).errorVectors_linearComponent(x))
  }

  /// Accumulates the adjoint (aka "transpose" or "dual") of `errorVectors` at `y` into `result`.
  ///
  /// - Requires: `storage` is the address of an `ArrayStorage` whose `Element` has a
  ///   subclass-specific `GaussianFactor` type.
  /// - Requires: `y.elementType == Element.ErrorVector.self`.
  /// - Requires: `result` contains elements at all of `self`'s elements' inputs.
  @_specialize(where Element == JacobianFactor3x3_1)
  @_specialize(where Element == JacobianFactor3x3_2)
  @_specialize(where Element == JacobianFactor6x6_1)
  @_specialize(where Element == JacobianFactor6x6_2)
  @_specialize(where Element == JacobianFactor9x3x3_1)
  @_specialize(where Element == JacobianFactor9x3x3_2)
  @_specialize(where Element == JacobianFactor12x6_1)
  @_specialize(where Element == JacobianFactor12x6_2)
  @_specialize(where Element == ScalarJacobianFactor<Vector1>)
  @_specialize(where Element == ScalarJacobianFactor<Vector2>)
  @_specialize(where Element == ScalarJacobianFactor<Vector3>)
  @_specialize(where Element == ScalarJacobianFactor<Vector4>)
  @_specialize(where Element == ScalarJacobianFactor<Vector5>)
  @_specialize(where Element == ScalarJacobianFactor<Vector6>)
  @_specialize(where Element == ScalarJacobianFactor<Vector6>)
  @_specialize(where Element == ScalarJacobianFactor<Vector7>)
  @_specialize(where Element == ScalarJacobianFactor<Vector8>)
  @_specialize(where Element == ScalarJacobianFactor<Vector9>)
  @_specialize(where Element == ScalarJacobianFactor<Vector10>)
  @_specialize(where Element == ScalarJacobianFactor<Vector11>)
  @_specialize(where Element == ScalarJacobianFactor<Vector12>)
  override class func errorVectors_linearComponent_adjoint(
    _ storage: UnsafeRawPointer,
    _ y: AnyElementArrayBuffer,
    into result: inout VariableAssignments
  ) {
    asStorage(storage).errorVectors_linearComponent_adjoint(
      .init(unsafelyDowncasting: y),
      into: &result
    )
  }
}

extension AnyArrayBuffer where Dispatch == GaussianFactorArrayDispatch {
  /// Creates an instance from a typed buffer of `Element`
  init<Element: GaussianFactor>(_ src: ArrayBuffer<Element>) {
    self.init(
      storage: src.storage,
      dispatch: GaussianFactorArrayDispatch_<Element>.self)
  }
}

extension AnyArrayBuffer where Dispatch: GaussianFactorArrayDispatch {
  /// Returns the error vectors, at `x`, of the factors.
  func errorVectors(at x: VariableAssignments) -> AnyVectorArrayBuffer {
    withUnsafePointer(to: storage) { dispatch.errorVectors($0, at: x) }
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
      dispatch.errorVectors_linearComponent_adjoint($0, y, into: &result)
    }
  }
}
