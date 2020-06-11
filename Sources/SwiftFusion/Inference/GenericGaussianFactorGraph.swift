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

public struct GenericGaussianFactorGraph {
  var inputZero: VariableAssignments
  var contiguousStorage: [ObjectIdentifier: AnyArrayBuffer<AnyGaussianFactorStorage>]

  func errorVectors(at x: VariableAssignments) -> VariableAssignments {
    return VariableAssignments(contiguousStorage: contiguousStorage.mapValues { factors in
      AnyArrayBuffer(factors.errorVectors(at: x))
    })
  }

  func errorVectorLinearComponents(at x: VariableAssignments) -> VariableAssignments {
    return VariableAssignments(contiguousStorage: contiguousStorage.mapValues { factors in
      AnyArrayBuffer(factors.linearForward(x))
    })
  }

  func errorVectorLinearComponentAdjoints(_ y: VariableAssignments) -> VariableAssignments {
    var x = inputZero
    contiguousStorage.forEach { (key, factor) in
      factor.linearAdjoint(y.contiguousStorage[key].unsafelyUnwrapped, into: &x)
    }
    return x
  }
}
