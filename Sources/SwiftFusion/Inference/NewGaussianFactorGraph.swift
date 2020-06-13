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

/// A factor graph whose factors are all `NewGaussianFactor`s.
public struct NewGaussianFactorGraph {
  /// Dictionary from factor type to contiguous storage for that type.
  var storage: [ObjectIdentifier: AnyArrayBuffer<AnyGaussianFactorStorage>]

  /// Assignment of zero to all the variables in the factor graph.
  var zeroValues: VariableAssignments

  /// For each variable, add a Jacobian factor that scales it by `scalar`.
  ///
  /// Precondition: `self` doesn't already contain scalar Jacobian factors. (Temporary limitation
  /// that we can remove when necessary.)
  public mutating func addScalarJacobians(_ scalar: Double) {
    zeroValues.storage.values.forEach { value in
      let (jacsKey, jacs) = value.cast(to: AnyVectorStorage.self)!.jacobians(scalar: scalar)
      // TODO: Support adding more jacobians of the same type.
      precondition(storage[jacsKey] == nil)
      storage[jacsKey] = jacs
    }
  }

  /// Returns the error vectors, at `x`, of all the factors.
  func errorVectors(at x: AllVectors) -> AllVectors {
    return AllVectors(storage: storage.mapValues { factors in
      AnyArrayBuffer(factors.errorVectors(at: x))
    })
  }

  /// Returns the linear component of `errorVectors`, evaluated at `x`.
  func errorVectors_linearComponent(at x: AllVectors) -> AllVectors {
    return AllVectors(storage: storage.mapValues { factors in
      AnyArrayBuffer(factors.errorVector_linearComponent(x))
    })
  }

  /// Returns the adjoint of the linear component of `errorVectors`, evaluated at `y`.
  func errorVectors_linearComponent_adjoint(_ y: AllVectors) -> AllVectors {
    var x = zeroValues
    storage.forEach { (key, factor) in
      factor.errorVector_linearComponent_adjoint(y.storage[key].unsafelyUnwrapped, into: &x)
    }
    return x
  }
}
