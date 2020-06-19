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

/// A factor graph.
public struct NewFactorGraph {
  /// Dictionary from factor type to contiguous storage for that type.
  var storage: [ObjectIdentifier: AnyArrayBuffer<AnyFactorStorage>] = [:]

  /// Creates an empty instance.
  public init() {}

  /// Creates an instance backed by the given `storage`.
  internal init(storage: [ObjectIdentifier: AnyArrayBuffer<AnyFactorStorage>]) {
    self.storage = storage
  }

  /// Stores `factor` in the graph.
  public mutating func store<T: NewFactor>(_ factor: T) {
    _ = storage[
      ObjectIdentifier(T.self),
      default: AnyArrayBuffer(ArrayBuffer<FactorArrayStorage<T>>())
    ].append(factor)
  }

  /// Stores `factor` in the graph.
  public mutating func store<T: NewLinearizableFactor>(_ factor: T) {
    _ = storage[
      ObjectIdentifier(T.self),
      default: AnyArrayBuffer(ArrayBuffer<LinearizableFactorArrayStorage<T>>())
    ].append(factor)
  }

  /// Stores `factor` in the graph.
  public mutating func store<T: NewGaussianFactor>(_ factor: T) {
    _ = storage[
      ObjectIdentifier(T.self),
      default: AnyArrayBuffer(ArrayBuffer<GaussianFactorArrayStorage<T>>())
    ].append(factor)
  }

  /// Returns the total error, at `x`, of all the factors.
  public func error(at x: VariableAssignments) -> Double {
    return storage.values.lazy.map { $0.errors(at: x).reduce(0, +) }.reduce(0, +)
  }

  /// Returns the total error, at `x`, of all the linearizable factors.
  public func linearizableError(at x: VariableAssignments) -> Double {
    return storage.values.reduce(0) { (result, factors) in
      guard let linearizableFactors = factors.cast(to: AnyLinearizableFactorStorage.self) else {
        return result
      }
      return result + linearizableFactors.errors(at: x).reduce(0, +)
    }
  }

  /// Returns the error vectors, at `x`, of all the linearizable factors.
  public func errorVectors(at x: VariableAssignments) -> AllVectors {
    return AllVectors(storage: storage.compactMapValues { factors in
      guard let linearizableFactors = factors.cast(to: AnyLinearizableFactorStorage.self) else {
        return nil
      }
      return AnyArrayBuffer(linearizableFactors.errorVectors(at: x))
    })
  }

  /// Returns a linear approximation to `self.errorVectors`, centered around `x`.
  ///
  /// The result has variables corresponding to each `Differentiable` variable in `self`. The
  /// The variable identified by `TypedID<T>(id)` in `self` corresponds to the variable identified
  /// by `TypedID<T.TangentVector>(id)` in the linear approximation.
  ///
  /// TODO: If there are different types of variables in `self` with the same tangent vector types,
  /// this leads to id clashes in `linearized(at: x)`. Fix this.
  ///
  /// The linear approximation satisfies the approximate equality:
  ///   `self.linearized(at: x).errorVectors(dx)` ≈ `self.errorVectors(at: x.moved(along: dx))`
  /// where the equality is exact when `dx == x.linearizedZero`.
  public func linearized(at x: VariableAssignments) -> NewGaussianFactorGraph {
    return NewGaussianFactorGraph(
      storage: storage.compactMapValues { factors in
        factors.cast(to: AnyLinearizableFactorStorage.self)?.linearized(at: x)
      },
      zeroValues: x.tangentVectorZeros
    )
  }
}
