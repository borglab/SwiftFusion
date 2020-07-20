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

protocol GaussianFactorGraphProtocol {
  associatedtype ErrorVectors
  associatedtype ErrorVectorsLinearComponentAdjoint
  associatedtype 
}

/// A factor graph whose factors are all `GaussianFactor`s.
public struct GaussianFactorGraph {
  private typealias Storage = TypewiseStorage<GaussianFactorArrayDispatch>
  
  /// Storage for all the graph's factors.
  private var storage: Storage

  /// Assignment of zero to all the variables in the factor graph.
  var zeroValues: AllVectors

  /// Creates an instance with the given `zeroValues`.
  public init(zeroValues: AllVectors) {
    self.storage = .init()
    self.zeroValues = zeroValues
  }

  /*
  /// Creates an instance with the given `storage` and `zeroValues`.
  init(
    storage: [ObjectIdentifier: AnyGaussianFactorArrayBuffer],
    zeroValues: AllVectors
  ) {
    self.storage = storage
    self.zeroValues = zeroValues
  }
   */
  
  /// Stores `factor` in the graph.
  public mutating func store<T: GaussianFactor>(_ factor: T) {
    storage.store(factor) { .init($0) }
  }

  /// For each variable, a Jacobian factor that scales it by `scaleFactor`.
  public func jacobians(scaleFactor: Double) -> TypeMappedStorage<GaussianFactorArrayDispatch> {
    zeroValues.map { $0.jacobians(scalar: scaleFactor) }
  }

  /// Returns the total error, at `x`, of all the factors.
  public func error(at x: VariableAssignments) -> Double {
    return 0.5 * errorVectors(at: x).squaredNorm
  }
  
  /// Returns the error vectors, at `x`, of all the factors.
  public func errorVectors(at x: AllVectors) -> AllVectors {
    return AllVectors(storage: storage.mapValues { factors in
      // Note: This is a safe upcast.
      AnyElementArrayBuffer(unsafelyCasting: factors.errorVectors(at: x))
    })
  }

  /// Returns the linear component of `errorVectors`, evaluated at `x`.
  public func errorVectors_linearComponent(at x: AllVectors) -> AllVectors {
    return AllVectors(storage: storage.mapValues { factors in
      // Note: This is a safe upcast.
      AnyElementArrayBuffer(unsafelyCasting: factors.errorVectors_linearComponent(x))
    })
  }

  /// Returns the adjoint of the linear component of `errorVectors`, evaluated at `y`.
  public func errorVectors_linearComponent_adjoint(_ y: AllVectors) -> AllVectors {
    var x = zeroValues
    storage.forEach { (key, factor) in
      factor.errorVectors_linearComponent_adjoint(y.storage[key].unsafelyUnwrapped, into: &x)
    }
    return x
  }
}
