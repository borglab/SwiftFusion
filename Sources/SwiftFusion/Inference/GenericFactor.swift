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

/// A factor in a factor graph.
///
/// Note: This is currently named `GenericFactor` to avoid clashing with the other `Factor`
/// protocol. When we completely replace `Factor`, we should rename this one to `Factor`.
protocol GenericFactor {
  /// A tuple of the variable types of variables adjacent to this factor.
  associatedtype Variables: VariableTuple

  /// The IDs of the variables adjacent to this factor.
  var edges: Variables.Indices { get }

  /// Returns the error given the values of the adjacent variables.
  func error(_: Variables) -> Double
}

// MARK: - `VariableTuple`.

/// Collections of variable types suitable for a factor.
protocol VariableTuple: TupleProtocol where Tail: VariableTuple {
  /// A tuple of `UnsafePointer`s to the types of the variables adjacent to a factor.
  associatedtype UnsafePointers: TupleProtocol

  /// A tuple of `TypedID`s referring to variables adjacent to a factor.
  associatedtype Indices: TupleProtocol

  /// Invokes `body` with the base addressess of the variable assignment buffers for the variable
  /// types in this `VariableTuble`.
  static func withVariableBufferBaseUnsafePointers<R>(
    _ variableAssignments: ValuesArray,
    _ body: (UnsafePointers) -> R
  ) -> R

  /// Creates a tuple containing the variable values in `variableBufferBases`, at `indices`.
  ///
  /// Parameter variableBufferBases: the base addresses from `withVariableBufferBaseUnsafePointers`.
  init(_ variableBufferBases: UnsafePointers, indices: Indices)
}

extension Empty: VariableTuple {
  typealias UnsafePointers = Self
  typealias Indices = Self

  static func withVariableBufferBaseUnsafePointers<R>(
    _ variableAssignments: ValuesArray,
    _ body: (UnsafePointers) -> R
  ) -> R {
    return body(Empty())
  }

  init(_ variableBufferBases: UnsafePointers, indices: Indices) {
    self.init()
  }
}

extension Tuple: VariableTuple where Tail: VariableTuple {
  typealias UnsafePointers = Tuple<UnsafePointer<Head>, Tail.UnsafePointers>
  typealias Indices = Tuple<TypedID<Head, Int>, Tail.Indices>

  static func withVariableBufferBaseUnsafePointers<R>(
    _ variableAssignments: ValuesArray,
    _ body: (UnsafePointers) -> R
  ) -> R {
    return variableAssignments
      .contiguousStorage[ObjectIdentifier(Head.self)].unsafelyUnwrapped
      .withUnsafeRawPointerToElements { headBase in
        return Tail.withVariableBufferBaseUnsafePointers(variableAssignments) { tailBase in
          return body(
            UnsafePointers(head: headBase.assumingMemoryBound(to: Head.self), tail: tailBase)
          )
        }
      }
  }

  init(_ variableBufferBases: UnsafePointers, indices: Indices) {
    self.init(
      head: variableBufferBases.head.advanced(by: indices.head.perTypeID).pointee,
      tail: Tail(variableBufferBases.tail, indices: indices.tail)
    )
  }
}
