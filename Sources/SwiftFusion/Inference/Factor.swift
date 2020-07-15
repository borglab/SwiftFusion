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
public protocol Factor {
  /// A tuple of the variable types of variables adjacent to this factor.
  associatedtype Variables: VariableTuple

  /// The IDs of the variables adjacent to this factor.
  var edges: Variables.Indices { get }

  /// Returns the error at `x`.
  ///
  /// This is typically interpreted as negative log-likelihood.
  func error(at x: Variables) -> Double
}

/// A factor whose `error` is a function of a vector-valued `errorVector` function.
public protocol VectorFactor: Factor {
  /// The type of the error vector.
  // TODO: Add a description of what an error vector is.
  associatedtype ErrorVector: EuclideanVectorN

  /// Returns the error vector at `x`.
  func errorVector(at x: Variables) -> ErrorVector

  /// A factor whose variables are the `Differentiable` subset of `Self`'s variables.
  associatedtype LinearizableComponent: LinearizableFactor
  where LinearizableComponent.ErrorVector == ErrorVector

  /// Returns the linearizable component of `self` at `x`, and returns the `Differentiable` subset
  /// of `x`.
  func linearizableComponent(at x: Variables)
    -> (LinearizableComponent, LinearizableComponent.Variables)
}

/// A factor whose `errorVector` function is linearizable with respect to all the variables.
public protocol LinearizableFactor: VectorFactor
where Variables: DifferentiableVariableTuple, Variables.TangentVector: EuclideanVectorN
{
  /// Returns the error vector given the values of the adjacent variables.
  @differentiable
  func errorVector(at x: Variables) -> ErrorVector
}

extension LinearizableFactor {
  public func linearizableComponent(at x: Variables) -> (Self, Variables) {
    return (self, x)
  }
}

/// Do not use this; it is a workaround for a Swift compiler limitation.
public protocol GaussianFactor_: Factor where Variables: EuclideanVectorN {}

/// A factor whose `errorVector` is a linear function of the variables, plus a constant.
public protocol GaussianFactor: LinearizableFactor, GaussianFactor_ {
  /// The linear component of `errorVector`.
  func errorVector_linearComponent(_ x: Variables) -> ErrorVector

  /// The adjoint (aka "transpose" or "dual") of the linear component of `errorVector`.
  func errorVector_linearComponent_adjoint(_ y: ErrorVector) -> Variables
}

/// A linear approximation of a linearizable factor.
public protocol LinearApproximationFactor: GaussianFactor {
  /// Creates a factor that linearly approximates `f` at `x`.
  init<F: LinearizableFactor>(linearizing f: F, at x: F.Variables)
  where F.Variables.TangentVector == Variables, F.ErrorVector == ErrorVector
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
  public typealias Indices = Tuple<TypedID<Head>, Tail.Indices>

  public static func withVariableBufferBaseUnsafePointers<R>(
    _ variableAssignments: VariableAssignments,
    _ body: (UnsafePointers) -> R
  ) -> R {
    let headArray = variableAssignments.storage[ObjectIdentifier(Head.self)].unsafelyUnwrapped
    return ArrayBuffer<Head>(unsafelyDowncasting: headArray)
      .withUnsafeBufferPointer { headBuffer in
        Tail.withVariableBufferBaseUnsafePointers(variableAssignments) { tailBase in
          body(
            UnsafePointers(
              head: headBuffer.baseAddress ?? UnsafePointer(bitPattern: -1)!,
              tail: tailBase
            )
          )
        }
      }
  }

  public static func ensureUniqueStorage(_ variableAssignments: inout VariableAssignments) {
    variableAssignments.storage[ObjectIdentifier(Head.self)]!.ensureUniqueStorage()
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
public protocol DifferentiableVariableTuple: Differentiable & VariableTuple
where TangentVector: DifferentiableVariableTuple {
  /// Returns the indices of the linearized variables corresponding to `indices` in the linearized
  /// factor graph.
  static func linearized(_ indices: Indices) -> TangentVector.Indices
}

extension Empty: DifferentiableVariableTuple {
  public static func linearized(_ indices: Indices) -> TangentVector.Indices {
    indices
  }
}

extension Tuple: DifferentiableVariableTuple
where Head: Differentiable, Tail: DifferentiableVariableTuple {
  public static func linearized(_ indices: Indices) -> TangentVector.Indices {
    TangentVector.Indices(
      head: TypedID<Head.TangentVector>(indices.head.perTypeID),
      tail: Tail.linearized(indices.tail)
    )
  }
}
