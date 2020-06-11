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
  /// Dictionary from variable type to contiguous storage for that type.
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

  public func error(at x: VariableAssignments) -> Double {
    return contiguousStorage.values.lazy.map { $0.errors(at: x).reduce(0, +) }.reduce(0, +)
  }

  public func linearized(at x: VariableAssignments) -> GenericGaussianFactorGraph {
    return GenericGaussianFactorGraph(
      inputZero: x.zeroTangent,
      contiguousStorage: contiguousStorage.compactMapValues { factors in
        factors.cast(to: AnyLinearizableFactorStorage.self)?.linearized(at: x)
      }
    )
  }
}
