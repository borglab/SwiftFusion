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

/// A factor graph whose factors are all `GenericGaussianFactor`s.
///
/// Note: This is currently named with a "Generic" prefix to avoid clashing with the other factors.
/// When we completely replace the existing factors with the "Generic" ones, we should remove this
/// prefix.
public struct GenericGaussianFactorGraph {
  /// Dictionary from factor type to contiguous storage for that type.
  var contiguousStorage: [ObjectIdentifier: AnyArrayBuffer<AnyGaussianFactorStorage>]

  /// Assignment of zero to all the variables in the factor graph.
  var zeroValues: VariableAssignments

  /// Returns the error vectors of all the factors, at `x`.
  ///
  /// Note: Using `VariableAssignments` as the return type is slightly inappropriate here because
  /// the return value assigns an error vector to each factor, not to each variable. We will fix
  /// this when we create a "Vectors" type for a heterogeneous collection of vectors.
  func errorVectors(at x: VariableAssignments) -> VariableAssignments {
    return VariableAssignments(contiguousStorage: contiguousStorage.mapValues { factors in
      AnyArrayBuffer(factors.errorVectors(at: x))
    })
  }

  /// Returns the linear component of the error vectors of all the factors, at `x`.
  ///
  /// Note: Using `VariableAssignments` as the return type is slightly inappropriate here because
  /// the return value assigns an error vector to each factor, not to each variable. We will fix
  /// this when we create a "Vectors" type for a heterogeneous collection of vectors.
  func errorVectorsLinearComponent(at x: VariableAssignments) -> VariableAssignments {
    return VariableAssignments(contiguousStorage: contiguousStorage.mapValues { factors in
      AnyArrayBuffer(factors.linearForward(x))
    })
  }

  /// Returns the adjoint of the linear component of the error vectors of all the factors, at `y`.
  ///
  /// Note: Using `VariableAssignments` as the argument type is slightly inappropriate here because
  /// the argument is an error vector for each factor, not for each variable. We will fix this when
  /// we create a "Vectors" type for a heterogeneous collection of vectors.
  func errorVectorsLinearComponentAdjoint(_ y: VariableAssignments) -> VariableAssignments {
    var x = zeroValues
    contiguousStorage.forEach { (key, factor) in
      factor.linearAdjoint(y.contiguousStorage[key].unsafelyUnwrapped, into: &x)
    }
    return x
  }
}
