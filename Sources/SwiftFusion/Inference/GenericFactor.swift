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
public protocol GenericFactor {
  /// A tuple of the variable types of variables adjacent to this factor.
  associatedtype Variables: VariableTuple

  /// The IDs of the variables adjacent to this factor.
  var edges: Variables.Indices { get }

  /// Returns the error given the values of the adjacent variables.
  func error(at x: Variables) -> Double
}

/// A factor with an error vector, which can be linearized around any point.
///
/// Note: This is currently named with a "Generic" prefix to avoid clashing with the other factors.
/// When we completely replace the existing factors with the "Generic" ones, we should remove this
/// prefix.
public protocol GenericLinearizableFactor: GenericFactor {
  /// The type of the error vector.
  // TODO: Add a description of what an error vector is.
  associatedtype ErrorVector: EuclideanVector

  /// Returns the error vector given the values of the adjacent variables.
  func errorVector(at x: Variables) -> ErrorVector

  /// The type of the linearized factor.
  associatedtype Linearization: GenericGaussianFactor

  /// Returns a factor whose `errorVector` linearly approximates `self`'s `errorVector` around the
  /// given point.
  func linearized(at x: Variables) -> Linearization
}

/// A factor with a vector error that is a linear function of the input, plus a constant.
///
/// Note: This is currently named with a "Generic" prefix to avoid clashing with the other factors.
/// When we completely replace the existing factors with the "Generic" ones, we should remove this
/// prefix.
public protocol GenericGaussianFactor: GenericLinearizableFactor where Variables: EuclideanVector {
  /// Returns the result of the linear function at the given point.
  func linearForward(_ x: Variables) -> ErrorVector

  /// Returns the result of the adjoint (aka "transpose" or "dual") of the linear function at the
  /// given point.
  func linearAdjoint(_ y: ErrorVector) -> Variables
}

// MARK: - `VariableTuple`.

/// Collections of variable types suitable for a factor.
public protocol VariableTuple: TupleProtocol where Tail: VariableTuple {
  /// A tuple of `UnsafePointer`s to the types of the variables adjacent to a factor.
  associatedtype UnsafePointers: TupleProtocol

  /// A tuple of `UnsafeMutablePointer`s to the types of the variables adjacent to a factor.
  associatedtype UnsafeMutablePointers: TupleProtocol

  /// A tuple of `TypedID`s referring to variables adjacent to a factor.
  associatedtype Indices: TupleProtocol

  /// Invokes `body` with the base addressess of the variable assignment buffers for the variable
  /// types in this `VariableTuble`.
  // TODO: `variableAssignments` should be the receiver.
  static func withVariableBufferBaseUnsafePointers<R>(
    _ variableAssignments: VariableAssignments,
    _ body: (UnsafePointers) -> R
  ) -> R

  /// Ensures that `variableAssignments` has unique storage for all the variable assignment buffers
  /// for the variable types in this `VariableTuble`.
  // TODO: `variableAssignments` should be the receiver.
  static func ensureUniqueStorage(_ variableAssignments: inout VariableAssignments)

  /// Returns mutable pointers referring to the same memory as `pointers`.
  // TODO: Maybe `pointers` should be the receiver.
  static func unsafeMutablePointers(mutating pointers: UnsafePointers) -> UnsafeMutablePointers

  /// Creates a tuple containing the variable values in `variableBufferBases`, at `indices`.
  ///
  /// Parameter variableBufferBases: the base addresses from `withVariableBufferBaseUnsafePointers`.
  init(_ variableBufferBases: UnsafeMutablePointers, indices: Indices)

  /// Stores `self` into `variableBufferBases`, at `indices`.
  func store(into variableBufferBases: UnsafeMutablePointers, indices: Indices)
}

extension VariableTuple {
  /// Invokes `body` with the base addressess of the variable assignment buffers for the variable
  /// types in this `VariableTuble`.
  // TODO: `variableAssignments` should be the receiver.
  static func withVariableBufferBaseUnsafeMutablePointers<R>(
    _ variableAssignments: inout VariableAssignments,
    _ body: (UnsafeMutablePointers) -> R
  ) -> R {
    ensureUniqueStorage(&variableAssignments)
    return withVariableBufferBaseUnsafePointers(variableAssignments) {
      body(unsafeMutablePointers(mutating: $0))
    }
  }

  /// Creates a tuple containing the variable values in `variableBufferBases`, at `indices`.
  ///
  /// Parameter variableBufferBases: the base addresses from `withVariableBufferBaseUnsafePointers`.
  init(_ variableBufferBases: UnsafePointers, indices: Indices) {
    self.init(Self.unsafeMutablePointers(mutating: variableBufferBases), indices: indices)
  }
}

extension Empty: VariableTuple {
  public typealias UnsafePointers = Self
  public typealias UnsafeMutablePointers = Self
  public typealias Indices = Self

  public static func withVariableBufferBaseUnsafePointers<R>(
    _ variableAssignments: VariableAssignments,
    _ body: (UnsafePointers) -> R
  ) -> R {
    return body(Empty())
  }

  public static func ensureUniqueStorage(_ variableAssignments: inout VariableAssignments) {}

  public static func unsafeMutablePointers(mutating pointers: UnsafePointers)
    -> UnsafeMutablePointers
  {
    Self()
  }

  public init(_ variableBufferBases: UnsafeMutablePointers, indices: Indices) {
    self.init()
  }

  public func store(into variableBufferBases: UnsafeMutablePointers, indices: Indices) {}
}

extension Tuple: VariableTuple where Tail: VariableTuple {
  public typealias UnsafePointers = Tuple<UnsafePointer<Head>, Tail.UnsafePointers>
  public typealias UnsafeMutablePointers =
    Tuple<UnsafeMutablePointer<Head>, Tail.UnsafeMutablePointers>
  public typealias Indices = Tuple<TypedID<Head, Int>, Tail.Indices>

  public static func withVariableBufferBaseUnsafePointers<R>(
    _ variableAssignments: VariableAssignments,
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

  public static func ensureUniqueStorage(_ variableAssignments: inout VariableAssignments) {
    variableAssignments.contiguousStorage[ObjectIdentifier(Head.self)]!.ensureUniqueStorage()
    Tail.ensureUniqueStorage(&variableAssignments)
  }

  public static func unsafeMutablePointers(mutating pointers: UnsafePointers)
    -> UnsafeMutablePointers
  {
    return UnsafeMutablePointers(
      head: UnsafeMutablePointer(mutating: pointers.head),
      tail: Tail.unsafeMutablePointers(mutating: pointers.tail)
    )
  }

  public init(_ variableBufferBases: UnsafeMutablePointers, indices: Indices) {
    self.init(
      head: variableBufferBases.head.advanced(by: indices.head.perTypeID).pointee,
      tail: Tail(variableBufferBases.tail, indices: indices.tail)
    )
  }

  public func store(into variableBufferBases: UnsafeMutablePointers, indices: Indices) {
    variableBufferBases.head.advanced(by: indices.head.perTypeID)
      .assign(repeating: self.head, count: 1)
    self.tail.store(into: variableBufferBases.tail, indices: indices.tail)
  }
}

/// Tuple of differentiable variable types suitable for a factor.
protocol DifferentiableVariableTuple: VariableTuple {
  /// A tuple of `TypedID`s referring to variables adjacent to a factor.
  associatedtype TangentIndices: TupleProtocol

  /// Returns the indices of the linearized variables corresponding to `indices` in the linearized
  /// factor graph.
  static func linearized(_ indices: Indices) -> TangentIndices
}

extension Empty: DifferentiableVariableTuple {
  typealias TangentIndices = Self
  static func linearized(_ indices: Indices) -> TangentIndices {
    indices
  }
}

extension Tuple: DifferentiableVariableTuple
where Head: Differentiable, Tail: DifferentiableVariableTuple {
  typealias TangentIndices = Tuple<TypedID<Head.TangentVector, Int>, Tail.TangentIndices>
  static func linearized(_ indices: Indices) -> TangentIndices {
    TangentIndices(
      head: TypedID<Head.TangentVector, Int>(indices.head.perTypeID),
      tail: Tail.linearized(indices.tail)
    )
  }
}
