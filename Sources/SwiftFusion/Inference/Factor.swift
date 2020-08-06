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
  /// A tuple type whose instances contain the values of the variables adjacent to this factor.
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
  associatedtype ErrorVector: Vector

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
where Variables: DifferentiableVariableTuple, Variables.TangentVector: Vector
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

/// A factor whose `errorVector` is a linear function of the variables, plus a constant.
public protocol GaussianFactor: LinearizableFactor
  where LinearizableComponent.Variables.TangentVector == Variables
{
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

/// The variable values adjacent to a factor.
public protocol VariableTuple: TupleProtocol where Tail: VariableTuple {
  /// `UnsafePointer`s to the base addresss of buffers containing values of the variable types in
  /// `Self`.
  associatedtype BufferBaseAddresses: TupleProtocol

  /// `UnsafeMutablePointer`s to the base addresss of buffers containing values of the variable
  /// types in `Self`.
  associatedtype MutableBufferBaseAddresses: TupleProtocol

  /// The indices used to address a `VariableTuple` scattered across per-type buffers.
  associatedtype Indices: TupleProtocol

  /// Invokes `body` with the base addressess of buffers from `v` storing
  /// instances of the variable types in `self`.
  // TODO: `v` should be the receiver.
  static func withMutableBufferBaseAddresses<R>(
    _ v: inout VariableAssignments,
    _ body: (MutableBufferBaseAddresses) -> R
  ) -> R

  /// Invokes `body` with the base addressess of buffers from `v` storing instances of the variable
  /// types in `self`.
  // TODO: `v` should be the receiver.
  static func withBufferBaseAddresses<R>(
    _ v: VariableAssignments,
    _ body: (BufferBaseAddresses) -> R
  ) -> R
  
  /// Returns a copy of `p` through which memory can't be mutated.
  // TODO: Maybe `pointers` should be the receiver.
  static func withoutMutation(_ p: MutableBufferBaseAddresses) -> BufferBaseAddresses

  /// Gathers the values at the given `positions` in the buffers having the given `baseAddresses`.
  init(at positions: Indices, in baseAddresses: BufferBaseAddresses)

  /// Scatters `self` into the given positions of buffers having the given `baseAddresses`.
  func assign(into positions: Indices, in baseAddresses: MutableBufferBaseAddresses)
}

extension Empty: VariableTuple {
  public typealias BufferBaseAddresses = Self
  public typealias MutableBufferBaseAddresses = Self
  public typealias Indices = Self

  public static func withMutableBufferBaseAddresses<R>(
    _ : inout VariableAssignments,
    _ body: (MutableBufferBaseAddresses) -> R
  ) -> R {
    return body(Empty())
  }

  public static func withBufferBaseAddresses<R>(
    _ : VariableAssignments,
    _ body: (BufferBaseAddresses) -> R
  ) -> R {
    return body(Empty())
  }

  public static func withoutMutation(_ p: MutableBufferBaseAddresses) -> BufferBaseAddresses {
    Self()
  }

  public init(at _: Indices, in _: MutableBufferBaseAddresses) {
    self.init()
  }

  public func assign(into _: Indices, in _: MutableBufferBaseAddresses) {}
}

extension Tuple: VariableTuple where Tail: VariableTuple {
  /// `UnsafePointer`s to the base addresss of buffers containing values of the variable types in
  /// `Self`.
  public typealias BufferBaseAddresses = Tuple<UnsafePointer<Head>, Tail.BufferBaseAddresses>
  
  /// `UnsafeMutablePointer`s to the base addresss of buffers containing values of the variable
  /// types in `Self`.
  public typealias MutableBufferBaseAddresses
    = Tuple<UnsafeMutablePointer<Head>, Tail.MutableBufferBaseAddresses>
  
  public typealias Indices = Tuple<TypedID<Head>, Tail.Indices>

  /// Invokes `body` with the base addressess of buffers from `v` storing instances of the variable
  /// types in `self`.
  public static func withMutableBufferBaseAddresses<R>(
    _ v: inout VariableAssignments,
    _ body: (MutableBufferBaseAddresses) -> R
  ) -> R {
    let headPointer = v.storage[
      existingKey: ObjectIdentifier(Head.self)
    ][
      existingElementType: Type<Head>()
    ].withUnsafeMutableBufferPointer { $0.baseAddress.unsafelyUnwrapped }

    return Tail.withMutableBufferBaseAddresses(&v) {
      body(.init(head: headPointer, tail: $0))
    }
  }

  /// Invokes `body` with the base addressess of buffers from `v` storing instances of the variable
  /// types in `self`.
  // TODO: `v` should be the receiver.
  public static func withBufferBaseAddresses<R>(
    _ v: VariableAssignments,
    _ body: (BufferBaseAddresses) -> R
  ) -> R {
    let headPointer = v.storage[
      existingKey: ObjectIdentifier(Head.self)
    ][
      existingElementType: Type<Head>()
    ].withUnsafeBufferPointer { $0.baseAddress.unsafelyUnwrapped }

    return Tail.withBufferBaseAddresses(v) {
      body(.init(head: headPointer, tail: $0))
    }
  }
  
  
  public static func withoutMutation(_ p: MutableBufferBaseAddresses) -> BufferBaseAddresses {
    return BufferBaseAddresses(
      head: .init(p.head),
      tail: Tail.withoutMutation(p.tail)
    )
  }

  /// Gathers the values at the given `positions` in the buffers having the given `baseAddresses`.
  public init(at positions: Indices, in baseAddresses: BufferBaseAddresses) {
    self.init(
      head: baseAddresses.head[positions.head.perTypeID],
      tail: Tail(at: positions.tail, in: baseAddresses.tail)
    )
  }

  /// Scatters `self` into the given positions of buffers having the given `baseAddresses`.
  public func assign(into positions: Indices, in variableBufferBases: MutableBufferBaseAddresses) {
    (variableBufferBases.head + positions.head.perTypeID)
      .assign(repeating: self.head, count: 1)
    self.tail.assign(into: positions.tail, in: variableBufferBases.tail)
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
