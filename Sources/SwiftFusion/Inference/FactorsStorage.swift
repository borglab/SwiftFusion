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
