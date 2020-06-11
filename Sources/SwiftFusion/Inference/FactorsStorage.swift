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

// MARK: - Storage for contiguous arrays of `GenericFactor`s.

/// Contiguous storage of homogeneous `Factor` values of statically unknown type.
class AnyFactorStorage: AnyArrayStorage {
  typealias FactorImplementation = AnyFactorStorageImplementation
  var factorImplementation: FactorImplementation {
    fatalError("implement me!")
  }

  /// Returns the errors of the factors given `variableAssignments`.
  final func errors(_ variableAssignments: ValuesArray) -> [Double] {
    factorImplementation.errors_(variableAssignments)
  }
}

extension AnyArrayBuffer where Storage: AnyFactorStorage {
  func errors(_ variableAssignments: ValuesArray) -> [Double] {
    storage.errors(variableAssignments)
  }
}

/// Contiguous storage of homogeneous `Factor` values of statically unknown type.
protocol AnyFactorStorageImplementation: AnyFactorStorage {
  /// Returns the errors of the factors given `variableAssignments`.
  func errors_(_ variableAssignments: ValuesArray) -> [Double]
}

/// APIs that depend on `Factor` `Element` type.
extension ArrayStorageImplementation where Element: GenericFactor {
  /// Returns the errors of the factors given `variableAssignments`.
  func errors_(_ variableAssignments: ValuesArray) -> [Double] {
    return Element.Variables.withVariableBufferBaseUnsafePointers(variableAssignments) { varsBuf in
      return withUnsafeMutableBufferPointer { factors in
        return factors.map { factor in
          return factor.error(Element.Variables(varsBuf, indices: factor.edges))
        }
      }
    }
  }
}

extension ArrayBuffer where Element: GenericFactor {
  func errors(_ variableAssignments: ValuesArray) -> [Double] {
    storage.errors_(variableAssignments)
  }
}

/// Type-erasable storage for contiguous `Factor` `Element` instances.
///
/// Note: instances have reference semantics.
final class FactorArrayStorage<Element: GenericFactor>:
  AnyFactorStorage, AnyFactorStorageImplementation,
  ArrayStorageImplementation
{
  override var implementation: AnyArrayStorageImplementation { self }
  override var factorImplementation: AnyFactorStorageImplementation { self }
}

// MARK: - Storage for contiguous arrays of `GenericLinearizableFactor`s.

/// Contiguous storage of homogeneous `LinearizableFactor` values of statically unknown type.
class AnyLinearizableFactorStorage: AnyFactorStorage {
  typealias LinearizableFactorImplementation = AnyLinearizableFactorStorageImplementation
  var linearizableFactorImplementation: LinearizableFactorImplementation {
    fatalError("implement me!")
  }

  /// Returns the error vectors of the factors given the values of the adjacent variables.
  func errorVectors(_ variableAssignments: ValuesArray) -> AnyArrayBuffer<AnyVectorStorage> {
    return linearizableFactorImplementation.errorVectors_(variableAssignments)
  }

  /// Returns the linearized factors at the given values.
  func linearized(_ variableAssignments: ValuesArray) -> AnyArrayBuffer<AnyGaussianFactorStorage> {
    return linearizableFactorImplementation.linearized_(variableAssignments)
  }
}

extension AnyArrayBuffer where Storage: AnyLinearizableFactorStorage {
  /// Returns the error vectors of the factors given the values of the adjacent variables.
  func errorVectors(_ variableAssignments: ValuesArray) -> AnyArrayBuffer<AnyVectorStorage> {
    return storage.errorVectors(variableAssignments)
  }

  /// Returns the linearized factors at the given values.
  func linearized(_ variableAssignments: ValuesArray) -> AnyArrayBuffer<AnyGaussianFactorStorage> {
    return storage.linearized(variableAssignments)
  }
}

/// Contiguous storage of homogeneous `LinearizableFactor` values of statically unknown type.
protocol AnyLinearizableFactorStorageImplementation: AnyLinearizableFactorStorage {
  /// Returns the error vectors of the factors given the values of the adjacent variables.
  func errorVectors_(_ variableAssignments: ValuesArray) -> AnyArrayBuffer<AnyVectorStorage>

  /// Returns the linearized factors at the given values.
  func linearized_(_ variableAssignments: ValuesArray) -> AnyArrayBuffer<AnyGaussianFactorStorage>
}

/// APIs that depend on `LinearizableFactor` `Element` type.
extension ArrayStorageImplementation where Element: GenericLinearizableFactor {
  /// Returns the error vectors of the factors given the values of the adjacent variables.
  func errorVectors(_ variableAssignments: ValuesArray)
    -> ArrayBuffer<VectorArrayStorage<Element.ErrorVector>>
  {
    Element.Variables.withVariableBufferBaseUnsafePointers(variableAssignments) { varsBufs in
      withUnsafeMutableBufferPointer { factors in
        ArrayBuffer(factors.lazy.map { factor in
          factor.errorVector(Element.Variables(varsBufs, indices: factor.edges))
        })
      }
    }
  }

  /// Returns the linearized factors at the given values.
  func linearized(_ variableAssignments: ValuesArray)
    -> ArrayBuffer<GaussianFactorArrayStorage<Element.Linearization>>
  {
    Element.Variables.withVariableBufferBaseUnsafePointers(variableAssignments) { varsBufs in
      withUnsafeMutableBufferPointer { factors in
        ArrayBuffer(factors.lazy.map { factor in
          factor.linearized(Element.Variables(varsBufs, indices: factor.edges))
        })
      }
    }
  }

  /// Returns the error vectors of the factors given the values of the adjacent variables.
  func errorVectors_(_ variableAssignments: ValuesArray) -> AnyArrayBuffer<AnyVectorStorage> {
    return AnyArrayBuffer(errorVectors(variableAssignments))
  }

  /// Returns the linearized factors at the given values.
  func linearized_(_ variableAssignments: ValuesArray) -> AnyArrayBuffer<AnyGaussianFactorStorage> {
    return AnyArrayBuffer(linearized(variableAssignments))
  }
}

extension ArrayBuffer where Element: GenericLinearizableFactor {
  /// Returns the error vectors of the factors given the values of the adjacent variables.
  func errorVectors(_ variableAssignments: ValuesArray)
    -> ArrayBuffer<VectorArrayStorage<Element.ErrorVector>>
  {
    storage.errorVectors(variableAssignments)
  }

  /// Returns the linearized factors at the given values.
  func linearized(_ variableAssignments: ValuesArray)
    -> ArrayBuffer<GaussianFactorArrayStorage<Element.Linearization>>
  {
    storage.linearized(variableAssignments)
  }
}

/// Type-erasable storage for contiguous `LinearizableFactor` `Element` instances.
///
/// Note: instances have reference semantics.
final class LinearizableFactorArrayStorage<Element: GenericLinearizableFactor>:
  AnyLinearizableFactorStorage, AnyLinearizableFactorStorageImplementation,
  ArrayStorageImplementation
{
  override var implementation: AnyArrayStorageImplementation { self }
  override var linearizableFactorImplementation: AnyLinearizableFactorStorageImplementation { self }
}

// MARK: - Storage for contiguous arrays of `GenericGaussianFactor`s.

/// Contiguous storage of homogeneous `GaussianFactor` values of statically unknown type.
class AnyGaussianFactorStorage: AnyFactorStorage {
  typealias GaussianFactorImplementation = AnyGaussianFactorStorageImplementation
  var gaussianFactorImplementation: GaussianFactorImplementation {
    fatalError("implement me!")
  }

  /// Returns the results of the factors' linear functions at the given point.
  func linearForward(_ variableAssignments: ValuesArray) -> AnyArrayBuffer<AnyVectorStorage> {
    gaussianFactorImplementation.linearForward_(variableAssignments)
  }

  /// Accumulates the adjoints (aka "transpose" or "dual") of the factors' linear functions at the
  /// given point into `result`.
  ///
  /// Precondition: `errorVectorsStart` points to memory with at least `count` initialized
  /// `Element.ErrorVector`s where `Element` is the element type of `self`.
  func linearAdjoint(_ errorVectorsStart: UnsafeRawPointer, into result: inout ValuesArray) {
    gaussianFactorImplementation.linearAdjoint_(errorVectorsStart, into: &result)
  }
}

extension AnyArrayBuffer where Storage: AnyGaussianFactorStorage {
  /// Returns the results of the factors' linear functions at the given point.
  func linearForward(_ variableAssignments: ValuesArray) -> AnyArrayBuffer<AnyVectorStorage> {
    return storage.linearForward(variableAssignments)
  }

  /// Accumulates the adjoints (aka "transpose" or "dual") of the factors' linear functions at the
  /// given point into `result`.
  ///
  /// Precondition: `errorVectorsStart` points to memory with at least `count` initialized
  /// `Element.ErrorVector`s where `Element` is the element type of `self`.
  func linearAdjoint(_ errorVectorsStart: UnsafeRawPointer, into result: inout ValuesArray) {
    storage.linearAdjoint(errorVectorsStart, into: &result)
  }
}

/// Contiguous storage of homogeneous `GaussianFactor` values of statically unknown type.
protocol AnyGaussianFactorStorageImplementation: AnyGaussianFactorStorage {
  /// Returns the results of the factors' linear functions at the given point.
  func linearForward_(_ variableAssignments: ValuesArray) -> AnyArrayBuffer<AnyVectorStorage>

  /// Accumulates the adjoints (aka "transpose" or "dual") of the factors' linear functions at the
  /// given point into `result`.
  ///
  /// Precondition: `errorVectorsStart` points to memory with at least `count` initialized
  /// `Element.ErrorVector`s where `Element` is the element type of `self`.
  func linearAdjoint_(_ errorVectorsStart: UnsafeRawPointer, into result: inout ValuesArray)
}

/// APIs that depend on `GaussianFactor` `Element` type.
extension ArrayStorageImplementation where Element: GenericGaussianFactor {
  /// Returns the results of the factors' linear functions at the given point.
  func linearForward(_ variableAssignments: ValuesArray)
    -> ArrayBuffer<VectorArrayStorage<Element.ErrorVector>>
  {
    Element.Variables.withVariableBufferBaseUnsafePointers(variableAssignments) { varsBufs in
      withUnsafeMutableBufferPointer { factors in
        ArrayBuffer(factors.map { factor in
          factor.linearForward(Element.Variables(varsBufs, indices: factor.edges))
        })
      }
    }
  }

  /// Accumulates the adjoints (aka "transpose" or "dual") of the factors' linear functions at the
  /// given point into `result`.
  ///
  /// Precondition: `errorVectorsStart` points to memory with at least `count` initialized
  /// `Element.ErrorVector`s.
  func linearAdjoint(
  _ errorVectorsStart: UnsafePointer<Element.ErrorVector>,
    into result: inout ValuesArray
  ) {
    Element.Variables.withVariableBufferBaseUnsafeMutablePointers(&result) { varsBufs in
      withUnsafeMutableBufferPointer { factors in
        factors.enumerated().forEach { (i, f) in
          let vars = Element.Variables(varsBufs, indices: f.edges)
          let newVars = vars + f.linearAdjoint(errorVectorsStart[i])
          newVars.store(into: varsBufs, indices: f.edges)
        }
      }
    }
  }

  /// Returns the results of the factors' linear functions at the given point.
  func linearForward_(_ variableAssignments: ValuesArray) -> AnyArrayBuffer<AnyVectorStorage> {
    return AnyArrayBuffer(linearForward(variableAssignments))
  }

  /// Accumulates the adjoints (aka "transpose" or "dual") of the factors' linear functions at the
  /// given point into `result`.
  ///
  /// Precondition: `errorVectorsStart` points to memory with at least `count` initialized
  /// `Element.ErrorVector`s.
  func linearAdjoint_(_ errorVectorsStart: UnsafeRawPointer, into result: inout ValuesArray) {
    linearAdjoint(
      errorVectorsStart.assumingMemoryBound(to: Element.ErrorVector.self),
      into: &result
    )
  }
}

extension ArrayBuffer where Element: GenericGaussianFactor {
  /// Returns the results of the factors' linear functions at the given point.
  func linearForward(_ variableAssignments: ValuesArray)
    -> ArrayBuffer<VectorArrayStorage<Element.ErrorVector>>
  {
    storage.linearForward(variableAssignments)
  }

  /// Accumulates the adjoints (aka "transpose" or "dual") of the factors' linear functions at the
  /// given point into `result`.
  ///
  /// Precondition: `errorVectorsStart` points to memory with at least `count` initialized
  /// `Element.ErrorVector`s.
  func linearAdjoint(
  _ errorVectorsStart: UnsafePointer<Element.ErrorVector>,
    into result: inout ValuesArray
  ) {
    storage.linearAdjoint(errorVectorsStart, into: &result)
  }
}

/// Type-erasable storage for contiguous `GaussianFactor` `Element` instances.
///
/// Note: instances have reference semantics.
final class GaussianFactorArrayStorage<Element: GenericGaussianFactor>:
  AnyGaussianFactorStorage, AnyGaussianFactorStorageImplementation,
  ArrayStorageImplementation
{
  override var implementation: AnyArrayStorageImplementation { self }
  override var gaussianFactorImplementation: AnyGaussianFactorStorageImplementation { self }
}
