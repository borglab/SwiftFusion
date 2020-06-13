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

// MARK: - Storage for contiguous arrays of `NewFactor`s.

/// Contiguous storage of homogeneous `Factor` values of statically unknown type.
class AnyFactorStorage: AnyArrayStorage {
  typealias FactorImplementation = AnyFactorStorageImplementation
  var factorImplementation: FactorImplementation {
    fatalError("implement me!")
  }

  /// Returns the errors of the factors given `x`.
  final func errors(at x: VariableAssignments) -> [Double] {
    factorImplementation.errors_(at: x)
  }
}

extension AnyArrayBuffer where Storage: AnyFactorStorage {
  func errors(at x: VariableAssignments) -> [Double] {
    storage.errors(at: x)
  }
}

/// Contiguous storage of homogeneous `Factor` values of statically unknown type.
protocol AnyFactorStorageImplementation: AnyFactorStorage {
  /// Returns the errors of the factors given `variableAssignments`.
  func errors_(at x: VariableAssignments) -> [Double]
}

/// APIs that depend on `Factor` `Element` type.
extension ArrayStorageImplementation where Element: NewFactor {
  /// Returns the errors of the factors given `variableAssignments`.
  func errors_(at x: VariableAssignments) -> [Double] {
    return Element.Variables.withVariableBufferBaseUnsafePointers(x) { varsBuf in
      return withUnsafeMutableBufferPointer { factors in
        return factors.map { factor in
          return factor.error(at: Element.Variables(varsBuf, indices: factor.edges))
        }
      }
    }
  }
}

extension ArrayBuffer where Element: NewFactor {
  func errors(at x: VariableAssignments) -> [Double] {
    storage.errors_(at: x)
  }
}

/// Type-erasable storage for contiguous `Factor` `Element` instances.
///
/// Note: instances have reference semantics.
final class FactorArrayStorage<Element: NewFactor>:
  AnyFactorStorage, AnyFactorStorageImplementation,
  ArrayStorageImplementation
{
  override var implementation: AnyArrayStorageImplementation { self }
  override var factorImplementation: AnyFactorStorageImplementation { self }
}

// MARK: - Storage for contiguous arrays of `NewLinearizableFactor`s.

/// Contiguous storage of homogeneous `LinearizableFactor` values of statically unknown type.
class AnyLinearizableFactorStorage: AnyFactorStorage {
  typealias LinearizableFactorImplementation = AnyLinearizableFactorStorageImplementation
  var linearizableFactorImplementation: LinearizableFactorImplementation {
    fatalError("implement me!")
  }

  /// Returns the error vectors of the factors given the values of the adjacent variables.
  func errorVectors(at x: VariableAssignments) -> AnyArrayBuffer<AnyVectorStorage> {
    return linearizableFactorImplementation.errorVectors_(at: x)
  }

  /// Returns the linearized factors at the given values.
  func linearized(at x: VariableAssignments) -> AnyArrayBuffer<AnyGaussianFactorStorage> {
    return linearizableFactorImplementation.linearized_(at: x)
  }
}

extension AnyArrayBuffer where Storage: AnyLinearizableFactorStorage {
  /// Returns the error vectors of the factors given the values of the adjacent variables.
  func errorVectors(at x: VariableAssignments) -> AnyArrayBuffer<AnyVectorStorage> {
    return storage.errorVectors(at: x)
  }

  /// Returns the linearized factors at the given values.
  func linearized(at x: VariableAssignments) -> AnyArrayBuffer<AnyGaussianFactorStorage> {
    return storage.linearized(at: x)
  }
}

/// Contiguous storage of homogeneous `LinearizableFactor` values of statically unknown type.
protocol AnyLinearizableFactorStorageImplementation: AnyLinearizableFactorStorage {
  /// Returns the error vectors of the factors given the values of the adjacent variables.
  func errorVectors_(at x: VariableAssignments) -> AnyArrayBuffer<AnyVectorStorage>

  /// Returns the linearized factors at the given values.
  func linearized_(at x: VariableAssignments) -> AnyArrayBuffer<AnyGaussianFactorStorage>
}

/// APIs that depend on `LinearizableFactor` `Element` type.
extension ArrayStorageImplementation where Element: NewLinearizableFactor {
  /// Returns the error vectors of the factors given the values of the adjacent variables.
  func errorVectors(at x: VariableAssignments)
    -> ArrayBuffer<VectorArrayStorage<Element.ErrorVector>>
  {
    Element.Variables.withVariableBufferBaseUnsafePointers(x) { varsBufs in
      withUnsafeMutableBufferPointer { factors in
        ArrayBuffer(factors.lazy.map { factor in
          factor.errorVector(at: Element.Variables(varsBufs, indices: factor.edges))
        })
      }
    }
  }

  /// Returns the linearized factors at the given values.
  func linearized(at x: VariableAssignments)
    -> ArrayBuffer<GaussianFactorArrayStorage<Element.Linearization>>
  {
    Element.Variables.withVariableBufferBaseUnsafePointers(x) { varsBufs in
      withUnsafeMutableBufferPointer { factors in
        ArrayBuffer(factors.lazy.map { factor in
          factor.linearized(at: Element.Variables(varsBufs, indices: factor.edges))
        })
      }
    }
  }

  /// Returns the error vectors of the factors given the values of the adjacent variables.
  func errorVectors_(at x: VariableAssignments) -> AnyArrayBuffer<AnyVectorStorage> {
    return AnyArrayBuffer(errorVectors(at: x))
  }

  /// Returns the linearized factors at the given values.
  func linearized_(at x: VariableAssignments) -> AnyArrayBuffer<AnyGaussianFactorStorage> {
    return AnyArrayBuffer(linearized(at: x))
  }
}

extension ArrayBuffer where Element: NewLinearizableFactor {
  /// Returns the error vectors of the factors given the values of the adjacent variables.
  func errorVectors(at x: VariableAssignments)
    -> ArrayBuffer<VectorArrayStorage<Element.ErrorVector>>
  {
    storage.errorVectors(at: x)
  }

  /// Returns the linearized factors at the given values.
  func linearized(at x: VariableAssignments)
    -> ArrayBuffer<GaussianFactorArrayStorage<Element.Linearization>>
  {
    storage.linearized(at: x)
  }
}

/// Type-erasable storage for contiguous `LinearizableFactor` `Element` instances.
///
/// Note: instances have reference semantics.
final class LinearizableFactorArrayStorage<Element: NewLinearizableFactor>:
  AnyLinearizableFactorStorage, AnyLinearizableFactorStorageImplementation,
  AnyFactorStorageImplementation, ArrayStorageImplementation
{
  override var implementation: AnyArrayStorageImplementation { self }
  override var factorImplementation: AnyFactorStorageImplementation { self }
  override var linearizableFactorImplementation: AnyLinearizableFactorStorageImplementation { self }
}

// MARK: - Storage for contiguous arrays of `NewGaussianFactor`s.

/// Contiguous storage of homogeneous `GaussianFactor` values of statically unknown type.
class AnyGaussianFactorStorage: AnyLinearizableFactorStorage {
  typealias GaussianFactorImplementation = AnyGaussianFactorStorageImplementation
  var gaussianFactorImplementation: GaussianFactorImplementation {
    fatalError("implement me!")
  }

  /// Returns the results of the factors' linear functions at the given point.
  func errorVector_linearComponent(_ x: VariableAssignments) -> AnyArrayBuffer<AnyVectorStorage> {
    gaussianFactorImplementation.errorVector_linearComponent_(x)
  }

  /// Accumulates the adjoints (aka "transpose" or "dual") of the factors' linear functions at the
  /// given point into `result`.
  ///
  /// Precondition: `errorVectorsStart` points to memory with at least `count` initialized
  /// `Element.ErrorVector`s where `Element` is the element type of `self`.
  func errorVector_linearComponent_adjoint(_ errorVectorsStart: UnsafeRawPointer, into result: inout VariableAssignments) {
    gaussianFactorImplementation.errorVector_linearComponent_adjoint_(errorVectorsStart, into: &result)
  }
}

extension AnyArrayBuffer where Storage: AnyGaussianFactorStorage {
  /// Returns the results of the factors' linear functions at the given point.
  func errorVector_linearComponent(_ x: VariableAssignments) -> AnyArrayBuffer<AnyVectorStorage> {
    return storage.errorVector_linearComponent(x)
  }

  /// Accumulates the adjoints (aka "transpose" or "dual") of the factors' linear functions at the
  /// given point into `result`.
  ///
  /// Precondition: `errorVectors` has at least `count` `Element.ErrorVector`s where `Element` is
  /// the element type of `self`.
  func errorVector_linearComponent_adjoint<VectorStorage>(
    _ errorVectors: AnyArrayBuffer<VectorStorage>,
    into result: inout VariableAssignments
  ) {
    errorVectors.withUnsafeRawPointerToElements { errorVectorsStart in
      storage.errorVector_linearComponent_adjoint(errorVectorsStart, into: &result)
    }
  }
}

/// Contiguous storage of homogeneous `GaussianFactor` values of statically unknown type.
protocol AnyGaussianFactorStorageImplementation: AnyGaussianFactorStorage {
  /// Returns the results of the factors' linear functions at the given point.
  func errorVector_linearComponent_(_ x: VariableAssignments) -> AnyArrayBuffer<AnyVectorStorage>

  /// Accumulates the adjoints (aka "transpose" or "dual") of the factors' linear functions at the
  /// given point into `result`.
  ///
  /// Precondition: `errorVectorsStart` points to memory with at least `count` initialized
  /// `Element.ErrorVector`s where `Element` is the element type of `self`.
  func errorVector_linearComponent_adjoint_(_ errorVectorsStart: UnsafeRawPointer, into result: inout VariableAssignments)
}

/// APIs that depend on `GaussianFactor` `Element` type.
extension ArrayStorageImplementation where Element: NewGaussianFactor {
  /// Returns the results of the factors' linear functions at the given point.
  func errorVector_linearComponent(_ x: VariableAssignments)
    -> ArrayBuffer<VectorArrayStorage<Element.ErrorVector>>
  {
    Element.Variables.withVariableBufferBaseUnsafePointers(x) { varsBufs in
      withUnsafeMutableBufferPointer { factors in
        ArrayBuffer(factors.map { factor in
          factor.errorVector_linearComponent(Element.Variables(varsBufs, indices: factor.edges))
        })
      }
    }
  }

  /// Accumulates the adjoints (aka "transpose" or "dual") of the factors' linear functions at the
  /// given point into `result`.
  ///
  /// Precondition: `errorVectors.count >= count`.
  func errorVector_linearComponent_adjoint<ErrorVectors: Collection>(
    _ errorVectors: ErrorVectors,
    into result: inout VariableAssignments
  ) where ErrorVectors.Element == Element.ErrorVector {
    Element.Variables.withVariableBufferBaseUnsafeMutablePointers(&result) { varsBufs in
      withUnsafeMutableBufferPointer { factors in
        zip(factors, errorVectors).forEach { (f, e) in
          let vars = Element.Variables(varsBufs, indices: f.edges)
          let newVars = vars + f.errorVector_linearComponent_adjoint(e)
          newVars.store(into: varsBufs, indices: f.edges)
        }
      }
    }
  }

  /// Returns the results of the factors' linear functions at the given point.
  // For reasonable performance, this must be specialized for all gaussian factor types that are
  // used in inner loops.
  @_specialize(where Self == GaussianFactorArrayStorage<ScalarJacobianFactor<Vector1>>)
  @_specialize(where Self == GaussianFactorArrayStorage<ScalarJacobianFactor<Vector2>>)
  @_specialize(where Self == GaussianFactorArrayStorage<ScalarJacobianFactor<Vector3>>)
  @_specialize(where Self == GaussianFactorArrayStorage<ScalarJacobianFactor<Vector4>>)
  @_specialize(where Self == GaussianFactorArrayStorage<ScalarJacobianFactor<Vector5>>)
  @_specialize(where Self == GaussianFactorArrayStorage<ScalarJacobianFactor<Vector6>>)
  @_specialize(where Self == GaussianFactorArrayStorage<JacobianFactor3x3_1>)
  @_specialize(where Self == GaussianFactorArrayStorage<JacobianFactor3x3_2>)
  @_specialize(where Self == GaussianFactorArrayStorage<JacobianFactor6x6_1>)
  @_specialize(where Self == GaussianFactorArrayStorage<JacobianFactor6x6_2>)
  func errorVector_linearComponent_(_ x: VariableAssignments) -> AnyArrayBuffer<AnyVectorStorage> {
    return AnyArrayBuffer(errorVector_linearComponent(x))
  }

  /// Accumulates the adjoints (aka "transpose" or "dual") of the factors' linear functions at the
  /// given point into `result`.
  ///
  /// Precondition: `errorVectorsStart` points to memory with at least `count` initialized
  /// `Element.ErrorVector`s.
  // For reasonable performance, this must be specialized for all gaussian factor types that are
  // used in inner loops.
  @_specialize(where Self == GaussianFactorArrayStorage<ScalarJacobianFactor<Vector1>>)
  @_specialize(where Self == GaussianFactorArrayStorage<ScalarJacobianFactor<Vector2>>)
  @_specialize(where Self == GaussianFactorArrayStorage<ScalarJacobianFactor<Vector3>>)
  @_specialize(where Self == GaussianFactorArrayStorage<ScalarJacobianFactor<Vector4>>)
  @_specialize(where Self == GaussianFactorArrayStorage<ScalarJacobianFactor<Vector5>>)
  @_specialize(where Self == GaussianFactorArrayStorage<ScalarJacobianFactor<Vector6>>)
  @_specialize(where Self == GaussianFactorArrayStorage<JacobianFactor3x3_1>)
  @_specialize(where Self == GaussianFactorArrayStorage<JacobianFactor3x3_2>)
  @_specialize(where Self == GaussianFactorArrayStorage<JacobianFactor6x6_1>)
  @_specialize(where Self == GaussianFactorArrayStorage<JacobianFactor6x6_2>)
  func errorVector_linearComponent_adjoint_(
    _ errorVectorsStart: UnsafeRawPointer,
    into result: inout VariableAssignments
  ) {
    errorVector_linearComponent_adjoint(
      UnsafeBufferPointer(
        start: errorVectorsStart.assumingMemoryBound(to: Element.ErrorVector.self),
        count: count
      ),
      into: &result
    )
  }
}

extension ArrayBuffer where Element: NewGaussianFactor {
  /// Returns the results of the factors' linear functions at the given point.
  func errorVector_linearComponent(_ x: VariableAssignments)
    -> ArrayBuffer<VectorArrayStorage<Element.ErrorVector>>
  {
    storage.errorVector_linearComponent(x)
  }

  /// Accumulates the adjoints (aka "transpose" or "dual") of the factors' linear functions at the
  /// given point into `result`.
  ///
  /// Precondition: `errorVectors.count >= count`.
  func errorVector_linearComponent_adjoint<ErrorVectors: Collection>(
    _ errorVectors: ErrorVectors,
    into result: inout VariableAssignments
  ) where ErrorVectors.Element == Element.ErrorVector {
    storage.errorVector_linearComponent_adjoint(errorVectors, into: &result)
  }
}

/// Type-erasable storage for contiguous `GaussianFactor` `Element` instances.
///
/// Note: instances have reference semantics.
final class GaussianFactorArrayStorage<Element: NewGaussianFactor>:
  AnyGaussianFactorStorage, AnyGaussianFactorStorageImplementation,
  AnyLinearizableFactorStorageImplementation, AnyFactorStorageImplementation,
  ArrayStorageImplementation
{
  override var implementation: AnyArrayStorageImplementation { self }
  override var factorImplementation: AnyFactorStorageImplementation { self }
  override var linearizableFactorImplementation: AnyLinearizableFactorStorageImplementation { self }
  override var gaussianFactorImplementation: AnyGaussianFactorStorageImplementation { self }
}
