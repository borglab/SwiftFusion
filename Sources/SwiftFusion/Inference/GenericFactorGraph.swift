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
///
/// Note: This is currently named with a "Generic" prefix to avoid clashing with the other factors.
/// When we completely replace the existing factors with the "Generic" ones, we should remove this
/// prefix.
public struct GenericFactorGraph {
  /// Dictionary from factor type to contiguous storage for that type.
  var contiguousStorage: [ObjectIdentifier: AnyArrayBuffer<AnyFactorStorage>] = [:]

  /// Creates an empty instance.
  public init() {}

  internal init(contiguousStorage: [ObjectIdentifier: AnyArrayBuffer<AnyFactorStorage>]) {
    self.contiguousStorage = contiguousStorage
  }

  /// Stores `factor` in the graph.
  public mutating func store<T: GenericFactor>(_ factor: T) {
    _ = contiguousStorage[
      ObjectIdentifier(T.self),
      default: AnyArrayBuffer(ArrayBuffer<FactorArrayStorage<T>>())
    ].append(factor)
  }

  /// Stores `factor` in the graph.
  public mutating func store<T: GenericLinearizableFactor>(_ factor: T) {
    _ = contiguousStorage[
      ObjectIdentifier(T.self),
      default: AnyArrayBuffer(ArrayBuffer<LinearizableFactorArrayStorage<T>>())
    ].append(factor)
  }

  /// Stores `factor` in the graph.
  public mutating func store<T: GenericGaussianFactor>(_ factor: T) {
    _ = contiguousStorage[
      ObjectIdentifier(T.self),
      default: AnyArrayBuffer(ArrayBuffer<GaussianFactorArrayStorage<T>>())
    ].append(factor)
  }

  /// Returns the total error of all the factors, at `x`.
  public func error(at x: VariableAssignments) -> Double {
    return contiguousStorage.values.lazy.map { $0.errors(at: x).reduce(0, +) }.reduce(0, +)
  }

  /// Returns the error vectors of all the linearizable factors, at `x`.
  ///
  /// Note: Using `VariableAssignments` as the return type is slightly inappropriate here because
  /// the return value assigns an error vector to each factor, not to each variable. We will fix
  /// this when we create a "Vectors" type for a heterogeneous collection of vectors.
  func errorVectors(at x: VariableAssignments) -> VariableAssignments {
    return VariableAssignments(contiguousStorage: contiguousStorage.compactMapValues { factors in
      guard let linearizableFactors = factors.cast(to: AnyLinearizableFactorStorage.self) else {
        return nil
      }
      return AnyArrayBuffer(linearizableFactors.errorVectors(at: x))
    })
  }

  /// Returns a linear approximation to `self`, centered around `x`.
  ///
  /// `linearized(at: x)` has one variable corresponding to each `Differentiable` variable in
  /// `self`. The variable identified by `TypedID<T>(id)` in `self` corresponds to the variable
  /// identified by `TypedID<T.TangentVector>(id)` in the linear approximation.
  ///
  /// TODO: If there are different types of variables in `self` with the same tangent vector types,
  /// this leads to id clashes in `linearized(at: x)`. Fix this.
  ///
  /// The linear approximation satisfies the approximate equality:
  ///   `linearized(at: x).errorVectors(dx) ~= self.errorVectors(at: x.moved(along: dx))`
  /// where the equality is exact when `dx == x.linearizedZero`.
  public func linearized(at x: VariableAssignments) -> GenericGaussianFactorGraph {
    return GenericGaussianFactorGraph(
      contiguousStorage: contiguousStorage.compactMapValues { factors in
        factors.cast(to: AnyLinearizableFactorStorage.self)?.linearized(at: x)
      },
      zeroValues: x.linearizedZero
    )
  }
}
