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

/// An identifier of a given abstract value with the value's type attached
///
/// - Parameter Value: the type of value this ID refers to.
///
/// Note: This is just a temporary placeholder until we get the real `TypedID` in penguin.
public typealias TypedID<Value> = TypewiseElementID<Value>

/// Assignments of values to factor graph variables.
public typealias VariableAssignments = TypewiseStorage<AnyObject>

extension VariableAssignments {
  /// Stores `value` as the assignment of a new variable, and returns the new variable's id.
  public mutating func store<T>(_ value: T) -> TypedID<T> {
    return self.store(value) { .init($0) }
  }

  /// Stores `value` as the assignment of a new variable, and returns the new variable's id.
  public mutating func store<T: Differentiable>(_ value: T) -> TypedID<T>
    where T.TangentVector: EuclideanVectorN
  {
    return self.store(value) { .init(AnyDifferentiableArrayBuffer($0)) }
  }

  /// Stores `value` as the assignment of a new variable, and returns the new variable's id.
  public mutating func store<T: EuclideanVectorN>(_ value: T) -> TypedID<T> {
    return self.store(value) { .init(AnyVectorArrayBuffer($0)) }
  }
}


/// Differentiable operations.
// TODO: There are some mutating operations here that copy, mutate, and write back. Make these
// more efficient.
extension VariableAssignments {
  public typealias TangentVectors = AllVectors
  
  /// For each differentiable value in `self`, the zero value of its tangent vector.
  public var tangentVectorZeros: TangentVectors {
    self.compactMap {
      AnyArrayBuffer<DifferentiableArrayDispatch>($0).map(\.tangentVectorZeros)
    }
  }

  /// Moves each differentiable variable along the corresponding element of `direction`.
  ///
  /// See `FactorGraph.linearized(at:)` for documentation about the correspondence between
  /// differentiable variables and their linearizations.
  public mutating func move(along direction: TangentVectors) {
    self.update(homomorphicArgument: direction) { v, d in
      v[dispatch: Type<DifferentiableArrayDispatch>()]?.move(along: d)
    }
  }

  /// See `move(along:)`.
  public func moved(along direction: TangentVectors) -> Self {
    // TODO: Make sure that this is efficient when we have a unique reference.
    var result = self
    result.move(along: direction)
    return result
  }
}
