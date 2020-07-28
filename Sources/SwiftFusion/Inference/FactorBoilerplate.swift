// WARNING: This is a generated file. Do not edit it. Instead, edit the corresponding ".gyb" file.
// See "generate.sh" in the root of this repository for instructions how to regenerate files.

// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 1)
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

/// Implements fixed-arity protocols `Factor1`, `VectorFactor1`, `LinearizableFactor1`, `Factor2`,
/// etc. It takes less boilerplate to conform to these fixed-arity protocols than it does to
/// conform to the arbitrary-arity protocols `Factor`, `VectorFactor`, and `LinearizableFactor`.

import PenguinStructures

// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 22)

// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 33)

// Artifact of Swift weakness.
/// Do not use this. Use `Factor1` instead.
public protocol Factor1_ {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 38)
  /// The 0-th variable type.
  associatedtype V0
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 41)

  /// Returns the error at the given point.
  ///
  /// This is typically interpreted as negative log-likelihood.
  func error(_: V0) -> Double
}

/// A factor in a factor graph.
public protocol Factor1: Factor, Factor1_
  where Variables == Tuple1<V0> {}

extension Factor1 {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 54)
  /// The variable vertex for this factor's 0-th variable.
  public var input0ID: TypedID<V0> { return edges.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 57)

  // Forwarding implementation.
  public func error(at x: Variables) -> Double {
    return error(x.head)
  }
}

// Artifact of Swift weakness.
/// Do not use this. Use `VectorFactor1` instead.
public protocol VectorFactor1_ {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 68)
  /// The 0-th variable type.
  associatedtype V0
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 71)

  typealias Variables = Tuple1<V0>

  associatedtype ErrorVector: EuclideanVector
  associatedtype LinearizableComponent: LinearizableFactor

  /// Returns the error vector at the given point.
  func errorVector(_: V0) -> ErrorVector

  /// Returns the linearizable component of `self` at the given point, and returns the
  /// `Differentiable` subset of the given variables.
  func linearizableComponent(_: V0)
    -> (LinearizableComponent, LinearizableComponent.Variables)
}

/// A factor whose `error` is a function of a vector-valued `errorVector` function.
public protocol VectorFactor1: VectorFactor, VectorFactor1_
  where Variables == Tuple1<V0> {}

extension VectorFactor1 {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 92)
  /// The variable vertex for this factor's 0-th variable.
  public var input0ID: TypedID<V0> { return edges.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 95)

  // Implements the error as half the squared norm of the error vector.
  public func error(at x: Variables) -> Double {
    return 0.5 * errorVector(at: x).squaredNorm
  }

  // Forwarding implementation.
  public func errorVector(at x: Variables) -> ErrorVector {
    return errorVector(x.head)
  }

  // Forwarding implementation.
  public func linearizableComponent(at x: Variables)
    -> (LinearizableComponent, LinearizableComponent.Variables)
  {
    return linearizableComponent(x.head)
  }
}

// Artifact of Swift weakness.
/// Do not use this. Use `LinearizableFactor1` instead.
public protocol LinearizableFactor1_ {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 118)
  /// The 0-th variable type.
  associatedtype V0: Differentiable
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 121)

  typealias Variables = Tuple1<V0>
  typealias LinearizableComponent = Self

  associatedtype ErrorVector: EuclideanVector

  /// Returns the error vector given the values of the adjacent variables.
  @differentiable
  func errorVector(_: V0) -> ErrorVector
}

/// A factor, with 1 variable(s), in a factor graph.
public protocol LinearizableFactor1: LinearizableFactor, LinearizableFactor1_
  where Variables == Tuple1<V0>, LinearizableComponent == Self {}

extension LinearizableFactor1 {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 138)
  /// The variable vertex for this factor's 0-th variable.
  public var input0ID: TypedID<V0> { return edges.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 141)

  // Implements the error as half the squared norm of the error vector.
  public func error(at x: Variables) -> Double {
    return 0.5 * errorVector(at: x).squaredNorm
  }

  // Forwarding implementation.
  @differentiable
  public func errorVector(at x: Variables) -> ErrorVector {
    return errorVector(x.head)
  }
}

// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 22)

// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 33)

// Artifact of Swift weakness.
/// Do not use this. Use `Factor2` instead.
public protocol Factor2_ {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 38)
  /// The 0-th variable type.
  associatedtype V0
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 38)
  /// The 1-th variable type.
  associatedtype V1
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 41)

  /// Returns the error at the given point.
  ///
  /// This is typically interpreted as negative log-likelihood.
  func error(_: V0, _: V1) -> Double
}

/// A factor in a factor graph.
public protocol Factor2: Factor, Factor2_
  where Variables == Tuple2<V0, V1> {}

extension Factor2 {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 54)
  /// The variable vertex for this factor's 0-th variable.
  public var input0ID: TypedID<V0> { return edges.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 54)
  /// The variable vertex for this factor's 1-th variable.
  public var input1ID: TypedID<V1> { return edges.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 57)

  // Forwarding implementation.
  public func error(at x: Variables) -> Double {
    return error(x.head, x.tail.head)
  }
}

// Artifact of Swift weakness.
/// Do not use this. Use `VectorFactor2` instead.
public protocol VectorFactor2_ {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 68)
  /// The 0-th variable type.
  associatedtype V0
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 68)
  /// The 1-th variable type.
  associatedtype V1
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 71)

  typealias Variables = Tuple2<V0, V1>

  associatedtype ErrorVector: EuclideanVector
  associatedtype LinearizableComponent: LinearizableFactor

  /// Returns the error vector at the given point.
  func errorVector(_: V0, _: V1) -> ErrorVector

  /// Returns the linearizable component of `self` at the given point, and returns the
  /// `Differentiable` subset of the given variables.
  func linearizableComponent(_: V0, _: V1)
    -> (LinearizableComponent, LinearizableComponent.Variables)
}

/// A factor whose `error` is a function of a vector-valued `errorVector` function.
public protocol VectorFactor2: VectorFactor, VectorFactor2_
  where Variables == Tuple2<V0, V1> {}

extension VectorFactor2 {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 92)
  /// The variable vertex for this factor's 0-th variable.
  public var input0ID: TypedID<V0> { return edges.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 92)
  /// The variable vertex for this factor's 1-th variable.
  public var input1ID: TypedID<V1> { return edges.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 95)

  // Implements the error as half the squared norm of the error vector.
  public func error(at x: Variables) -> Double {
    return 0.5 * errorVector(at: x).squaredNorm
  }

  // Forwarding implementation.
  public func errorVector(at x: Variables) -> ErrorVector {
    return errorVector(x.head, x.tail.head)
  }

  // Forwarding implementation.
  public func linearizableComponent(at x: Variables)
    -> (LinearizableComponent, LinearizableComponent.Variables)
  {
    return linearizableComponent(x.head, x.tail.head)
  }
}

// Artifact of Swift weakness.
/// Do not use this. Use `LinearizableFactor2` instead.
public protocol LinearizableFactor2_ {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 118)
  /// The 0-th variable type.
  associatedtype V0: Differentiable
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 118)
  /// The 1-th variable type.
  associatedtype V1: Differentiable
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 121)

  typealias Variables = Tuple2<V0, V1>
  typealias LinearizableComponent = Self

  associatedtype ErrorVector: EuclideanVector

  /// Returns the error vector given the values of the adjacent variables.
  @differentiable
  func errorVector(_: V0, _: V1) -> ErrorVector
}

/// A factor, with 2 variable(s), in a factor graph.
public protocol LinearizableFactor2: LinearizableFactor, LinearizableFactor2_
  where Variables == Tuple2<V0, V1>, LinearizableComponent == Self {}

extension LinearizableFactor2 {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 138)
  /// The variable vertex for this factor's 0-th variable.
  public var input0ID: TypedID<V0> { return edges.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 138)
  /// The variable vertex for this factor's 1-th variable.
  public var input1ID: TypedID<V1> { return edges.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 141)

  // Implements the error as half the squared norm of the error vector.
  public func error(at x: Variables) -> Double {
    return 0.5 * errorVector(at: x).squaredNorm
  }

  // Forwarding implementation.
  @differentiable
  public func errorVector(at x: Variables) -> ErrorVector {
    return errorVector(x.head, x.tail.head)
  }
}

// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 22)

// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 33)

// Artifact of Swift weakness.
/// Do not use this. Use `Factor3` instead.
public protocol Factor3_ {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 38)
  /// The 0-th variable type.
  associatedtype V0
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 38)
  /// The 1-th variable type.
  associatedtype V1
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 38)
  /// The 2-th variable type.
  associatedtype V2
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 41)

  /// Returns the error at the given point.
  ///
  /// This is typically interpreted as negative log-likelihood.
  func error(_: V0, _: V1, _: V2) -> Double
}

/// A factor in a factor graph.
public protocol Factor3: Factor, Factor3_
  where Variables == Tuple3<V0, V1, V2> {}

extension Factor3 {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 54)
  /// The variable vertex for this factor's 0-th variable.
  public var input0ID: TypedID<V0> { return edges.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 54)
  /// The variable vertex for this factor's 1-th variable.
  public var input1ID: TypedID<V1> { return edges.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 54)
  /// The variable vertex for this factor's 2-th variable.
  public var input2ID: TypedID<V2> { return edges.tail.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 57)

  // Forwarding implementation.
  public func error(at x: Variables) -> Double {
    return error(x.head, x.tail.head, x.tail.tail.head)
  }
}

// Artifact of Swift weakness.
/// Do not use this. Use `VectorFactor3` instead.
public protocol VectorFactor3_ {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 68)
  /// The 0-th variable type.
  associatedtype V0
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 68)
  /// The 1-th variable type.
  associatedtype V1
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 68)
  /// The 2-th variable type.
  associatedtype V2
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 71)

  typealias Variables = Tuple3<V0, V1, V2>

  associatedtype ErrorVector: EuclideanVector
  associatedtype LinearizableComponent: LinearizableFactor

  /// Returns the error vector at the given point.
  func errorVector(_: V0, _: V1, _: V2) -> ErrorVector

  /// Returns the linearizable component of `self` at the given point, and returns the
  /// `Differentiable` subset of the given variables.
  func linearizableComponent(_: V0, _: V1, _: V2)
    -> (LinearizableComponent, LinearizableComponent.Variables)
}

/// A factor whose `error` is a function of a vector-valued `errorVector` function.
public protocol VectorFactor3: VectorFactor, VectorFactor3_
  where Variables == Tuple3<V0, V1, V2> {}

extension VectorFactor3 {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 92)
  /// The variable vertex for this factor's 0-th variable.
  public var input0ID: TypedID<V0> { return edges.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 92)
  /// The variable vertex for this factor's 1-th variable.
  public var input1ID: TypedID<V1> { return edges.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 92)
  /// The variable vertex for this factor's 2-th variable.
  public var input2ID: TypedID<V2> { return edges.tail.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 95)

  // Implements the error as half the squared norm of the error vector.
  public func error(at x: Variables) -> Double {
    return 0.5 * errorVector(at: x).squaredNorm
  }

  // Forwarding implementation.
  public func errorVector(at x: Variables) -> ErrorVector {
    return errorVector(x.head, x.tail.head, x.tail.tail.head)
  }

  // Forwarding implementation.
  public func linearizableComponent(at x: Variables)
    -> (LinearizableComponent, LinearizableComponent.Variables)
  {
    return linearizableComponent(x.head, x.tail.head, x.tail.tail.head)
  }
}

// Artifact of Swift weakness.
/// Do not use this. Use `LinearizableFactor3` instead.
public protocol LinearizableFactor3_ {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 118)
  /// The 0-th variable type.
  associatedtype V0: Differentiable
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 118)
  /// The 1-th variable type.
  associatedtype V1: Differentiable
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 118)
  /// The 2-th variable type.
  associatedtype V2: Differentiable
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 121)

  typealias Variables = Tuple3<V0, V1, V2>
  typealias LinearizableComponent = Self

  associatedtype ErrorVector: EuclideanVector

  /// Returns the error vector given the values of the adjacent variables.
  @differentiable
  func errorVector(_: V0, _: V1, _: V2) -> ErrorVector
}

/// A factor, with 3 variable(s), in a factor graph.
public protocol LinearizableFactor3: LinearizableFactor, LinearizableFactor3_
  where Variables == Tuple3<V0, V1, V2>, LinearizableComponent == Self {}

extension LinearizableFactor3 {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 138)
  /// The variable vertex for this factor's 0-th variable.
  public var input0ID: TypedID<V0> { return edges.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 138)
  /// The variable vertex for this factor's 1-th variable.
  public var input1ID: TypedID<V1> { return edges.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 138)
  /// The variable vertex for this factor's 2-th variable.
  public var input2ID: TypedID<V2> { return edges.tail.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 141)

  // Implements the error as half the squared norm of the error vector.
  public func error(at x: Variables) -> Double {
    return 0.5 * errorVector(at: x).squaredNorm
  }

  // Forwarding implementation.
  @differentiable
  public func errorVector(at x: Variables) -> ErrorVector {
    return errorVector(x.head, x.tail.head, x.tail.tail.head)
  }
}

// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 22)

// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 33)

// Artifact of Swift weakness.
/// Do not use this. Use `Factor4` instead.
public protocol Factor4_ {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 38)
  /// The 0-th variable type.
  associatedtype V0
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 38)
  /// The 1-th variable type.
  associatedtype V1
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 38)
  /// The 2-th variable type.
  associatedtype V2
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 38)
  /// The 3-th variable type.
  associatedtype V3
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 41)

  /// Returns the error at the given point.
  ///
  /// This is typically interpreted as negative log-likelihood.
  func error(_: V0, _: V1, _: V2, _: V3) -> Double
}

/// A factor in a factor graph.
public protocol Factor4: Factor, Factor4_
  where Variables == Tuple4<V0, V1, V2, V3> {}

extension Factor4 {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 54)
  /// The variable vertex for this factor's 0-th variable.
  public var input0ID: TypedID<V0> { return edges.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 54)
  /// The variable vertex for this factor's 1-th variable.
  public var input1ID: TypedID<V1> { return edges.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 54)
  /// The variable vertex for this factor's 2-th variable.
  public var input2ID: TypedID<V2> { return edges.tail.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 54)
  /// The variable vertex for this factor's 3-th variable.
  public var input3ID: TypedID<V3> { return edges.tail.tail.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 57)

  // Forwarding implementation.
  public func error(at x: Variables) -> Double {
    return error(x.head, x.tail.head, x.tail.tail.head, x.tail.tail.tail.head)
  }
}

// Artifact of Swift weakness.
/// Do not use this. Use `VectorFactor4` instead.
public protocol VectorFactor4_ {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 68)
  /// The 0-th variable type.
  associatedtype V0
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 68)
  /// The 1-th variable type.
  associatedtype V1
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 68)
  /// The 2-th variable type.
  associatedtype V2
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 68)
  /// The 3-th variable type.
  associatedtype V3
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 71)

  typealias Variables = Tuple4<V0, V1, V2, V3>

  associatedtype ErrorVector: EuclideanVector
  associatedtype LinearizableComponent: LinearizableFactor

  /// Returns the error vector at the given point.
  func errorVector(_: V0, _: V1, _: V2, _: V3) -> ErrorVector

  /// Returns the linearizable component of `self` at the given point, and returns the
  /// `Differentiable` subset of the given variables.
  func linearizableComponent(_: V0, _: V1, _: V2, _: V3)
    -> (LinearizableComponent, LinearizableComponent.Variables)
}

/// A factor whose `error` is a function of a vector-valued `errorVector` function.
public protocol VectorFactor4: VectorFactor, VectorFactor4_
  where Variables == Tuple4<V0, V1, V2, V3> {}

extension VectorFactor4 {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 92)
  /// The variable vertex for this factor's 0-th variable.
  public var input0ID: TypedID<V0> { return edges.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 92)
  /// The variable vertex for this factor's 1-th variable.
  public var input1ID: TypedID<V1> { return edges.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 92)
  /// The variable vertex for this factor's 2-th variable.
  public var input2ID: TypedID<V2> { return edges.tail.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 92)
  /// The variable vertex for this factor's 3-th variable.
  public var input3ID: TypedID<V3> { return edges.tail.tail.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 95)

  // Implements the error as half the squared norm of the error vector.
  public func error(at x: Variables) -> Double {
    return 0.5 * errorVector(at: x).squaredNorm
  }

  // Forwarding implementation.
  public func errorVector(at x: Variables) -> ErrorVector {
    return errorVector(x.head, x.tail.head, x.tail.tail.head, x.tail.tail.tail.head)
  }

  // Forwarding implementation.
  public func linearizableComponent(at x: Variables)
    -> (LinearizableComponent, LinearizableComponent.Variables)
  {
    return linearizableComponent(x.head, x.tail.head, x.tail.tail.head, x.tail.tail.tail.head)
  }
}

// Artifact of Swift weakness.
/// Do not use this. Use `LinearizableFactor4` instead.
public protocol LinearizableFactor4_ {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 118)
  /// The 0-th variable type.
  associatedtype V0: Differentiable
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 118)
  /// The 1-th variable type.
  associatedtype V1: Differentiable
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 118)
  /// The 2-th variable type.
  associatedtype V2: Differentiable
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 118)
  /// The 3-th variable type.
  associatedtype V3: Differentiable
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 121)

  typealias Variables = Tuple4<V0, V1, V2, V3>
  typealias LinearizableComponent = Self

  associatedtype ErrorVector: EuclideanVector

  /// Returns the error vector given the values of the adjacent variables.
  @differentiable
  func errorVector(_: V0, _: V1, _: V2, _: V3) -> ErrorVector
}

/// A factor, with 4 variable(s), in a factor graph.
public protocol LinearizableFactor4: LinearizableFactor, LinearizableFactor4_
  where Variables == Tuple4<V0, V1, V2, V3>, LinearizableComponent == Self {}

extension LinearizableFactor4 {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 138)
  /// The variable vertex for this factor's 0-th variable.
  public var input0ID: TypedID<V0> { return edges.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 138)
  /// The variable vertex for this factor's 1-th variable.
  public var input1ID: TypedID<V1> { return edges.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 138)
  /// The variable vertex for this factor's 2-th variable.
  public var input2ID: TypedID<V2> { return edges.tail.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 138)
  /// The variable vertex for this factor's 3-th variable.
  public var input3ID: TypedID<V3> { return edges.tail.tail.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 141)

  // Implements the error as half the squared norm of the error vector.
  public func error(at x: Variables) -> Double {
    return 0.5 * errorVector(at: x).squaredNorm
  }

  // Forwarding implementation.
  @differentiable
  public func errorVector(at x: Variables) -> ErrorVector {
    return errorVector(x.head, x.tail.head, x.tail.tail.head, x.tail.tail.tail.head)
  }
}

// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 22)

// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 33)

// Artifact of Swift weakness.
/// Do not use this. Use `Factor5` instead.
public protocol Factor5_ {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 38)
  /// The 0-th variable type.
  associatedtype V0
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 38)
  /// The 1-th variable type.
  associatedtype V1
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 38)
  /// The 2-th variable type.
  associatedtype V2
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 38)
  /// The 3-th variable type.
  associatedtype V3
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 38)
  /// The 4-th variable type.
  associatedtype V4
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 41)

  /// Returns the error at the given point.
  ///
  /// This is typically interpreted as negative log-likelihood.
  func error(_: V0, _: V1, _: V2, _: V3, _: V4) -> Double
}

/// A factor in a factor graph.
public protocol Factor5: Factor, Factor5_
  where Variables == Tuple5<V0, V1, V2, V3, V4> {}

extension Factor5 {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 54)
  /// The variable vertex for this factor's 0-th variable.
  public var input0ID: TypedID<V0> { return edges.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 54)
  /// The variable vertex for this factor's 1-th variable.
  public var input1ID: TypedID<V1> { return edges.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 54)
  /// The variable vertex for this factor's 2-th variable.
  public var input2ID: TypedID<V2> { return edges.tail.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 54)
  /// The variable vertex for this factor's 3-th variable.
  public var input3ID: TypedID<V3> { return edges.tail.tail.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 54)
  /// The variable vertex for this factor's 4-th variable.
  public var input4ID: TypedID<V4> { return edges.tail.tail.tail.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 57)

  // Forwarding implementation.
  public func error(at x: Variables) -> Double {
    return error(x.head, x.tail.head, x.tail.tail.head, x.tail.tail.tail.head, x.tail.tail.tail.tail.head)
  }
}

// Artifact of Swift weakness.
/// Do not use this. Use `VectorFactor5` instead.
public protocol VectorFactor5_ {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 68)
  /// The 0-th variable type.
  associatedtype V0
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 68)
  /// The 1-th variable type.
  associatedtype V1
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 68)
  /// The 2-th variable type.
  associatedtype V2
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 68)
  /// The 3-th variable type.
  associatedtype V3
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 68)
  /// The 4-th variable type.
  associatedtype V4
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 71)

  typealias Variables = Tuple5<V0, V1, V2, V3, V4>

  associatedtype ErrorVector: EuclideanVector
  associatedtype LinearizableComponent: LinearizableFactor

  /// Returns the error vector at the given point.
  func errorVector(_: V0, _: V1, _: V2, _: V3, _: V4) -> ErrorVector

  /// Returns the linearizable component of `self` at the given point, and returns the
  /// `Differentiable` subset of the given variables.
  func linearizableComponent(_: V0, _: V1, _: V2, _: V3, _: V4)
    -> (LinearizableComponent, LinearizableComponent.Variables)
}

/// A factor whose `error` is a function of a vector-valued `errorVector` function.
public protocol VectorFactor5: VectorFactor, VectorFactor5_
  where Variables == Tuple5<V0, V1, V2, V3, V4> {}

extension VectorFactor5 {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 92)
  /// The variable vertex for this factor's 0-th variable.
  public var input0ID: TypedID<V0> { return edges.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 92)
  /// The variable vertex for this factor's 1-th variable.
  public var input1ID: TypedID<V1> { return edges.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 92)
  /// The variable vertex for this factor's 2-th variable.
  public var input2ID: TypedID<V2> { return edges.tail.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 92)
  /// The variable vertex for this factor's 3-th variable.
  public var input3ID: TypedID<V3> { return edges.tail.tail.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 92)
  /// The variable vertex for this factor's 4-th variable.
  public var input4ID: TypedID<V4> { return edges.tail.tail.tail.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 95)

  // Implements the error as half the squared norm of the error vector.
  public func error(at x: Variables) -> Double {
    return 0.5 * errorVector(at: x).squaredNorm
  }

  // Forwarding implementation.
  public func errorVector(at x: Variables) -> ErrorVector {
    return errorVector(x.head, x.tail.head, x.tail.tail.head, x.tail.tail.tail.head, x.tail.tail.tail.tail.head)
  }

  // Forwarding implementation.
  public func linearizableComponent(at x: Variables)
    -> (LinearizableComponent, LinearizableComponent.Variables)
  {
    return linearizableComponent(x.head, x.tail.head, x.tail.tail.head, x.tail.tail.tail.head, x.tail.tail.tail.tail.head)
  }
}

// Artifact of Swift weakness.
/// Do not use this. Use `LinearizableFactor5` instead.
public protocol LinearizableFactor5_ {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 118)
  /// The 0-th variable type.
  associatedtype V0: Differentiable
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 118)
  /// The 1-th variable type.
  associatedtype V1: Differentiable
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 118)
  /// The 2-th variable type.
  associatedtype V2: Differentiable
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 118)
  /// The 3-th variable type.
  associatedtype V3: Differentiable
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 118)
  /// The 4-th variable type.
  associatedtype V4: Differentiable
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 121)

  typealias Variables = Tuple5<V0, V1, V2, V3, V4>
  typealias LinearizableComponent = Self

  associatedtype ErrorVector: EuclideanVector

  /// Returns the error vector given the values of the adjacent variables.
  @differentiable
  func errorVector(_: V0, _: V1, _: V2, _: V3, _: V4) -> ErrorVector
}

/// A factor, with 5 variable(s), in a factor graph.
public protocol LinearizableFactor5: LinearizableFactor, LinearizableFactor5_
  where Variables == Tuple5<V0, V1, V2, V3, V4>, LinearizableComponent == Self {}

extension LinearizableFactor5 {
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 138)
  /// The variable vertex for this factor's 0-th variable.
  public var input0ID: TypedID<V0> { return edges.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 138)
  /// The variable vertex for this factor's 1-th variable.
  public var input1ID: TypedID<V1> { return edges.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 138)
  /// The variable vertex for this factor's 2-th variable.
  public var input2ID: TypedID<V2> { return edges.tail.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 138)
  /// The variable vertex for this factor's 3-th variable.
  public var input3ID: TypedID<V3> { return edges.tail.tail.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 138)
  /// The variable vertex for this factor's 4-th variable.
  public var input4ID: TypedID<V4> { return edges.tail.tail.tail.tail.head }
// ###sourceLocation(file: "Sources/SwiftFusion/Inference/FactorBoilerplate.swift.gyb", line: 141)

  // Implements the error as half the squared norm of the error vector.
  public func error(at x: Variables) -> Double {
    return 0.5 * errorVector(at: x).squaredNorm
  }

  // Forwarding implementation.
  @differentiable
  public func errorVector(at x: Variables) -> ErrorVector {
    return errorVector(x.head, x.tail.head, x.tail.tail.head, x.tail.tail.tail.head, x.tail.tail.tail.tail.head)
  }
}

