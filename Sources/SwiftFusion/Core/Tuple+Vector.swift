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

extension Empty: AdditiveArithmetic {
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    return Empty()
  }
  public static func - (_ lhs: Self, _ rhs: Self) -> Self {
    return Empty()
  }
  public static var zero: Self {
    return Empty()
  }
}

extension Tuple: AdditiveArithmetic where Head: AdditiveArithmetic, Tail: AdditiveArithmetic {
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    return Tuple(head: lhs.head + rhs.head, tail: lhs.tail + rhs.tail)
  }
  public static func - (_ lhs: Self, _ rhs: Self) -> Self {
    return Tuple(head: lhs.head - rhs.head, tail: lhs.tail - rhs.tail)
  }
  public static var zero: Self {
    return Tuple(head: Head.zero, tail: Tail.zero)
  }
}

extension Empty: Differentiable {
  public typealias TangentVector = Self
  public mutating func move(along direction: TangentVector) {}
  
  public var zeroTangentVectorInitializer: () -> TangentVector {
    { .zero }
  }
}

extension Tuple: Differentiable
where Head: Differentiable, Tail: Differentiable, Tail.TangentVector: TupleProtocol {
  public typealias TangentVector = Tuple<Head.TangentVector, Tail.TangentVector>
  public mutating func move(along direction: TangentVector) {
    head.move(along: direction.head)
    tail.move(along: direction.tail)
  }
  public var zeroTangentVectorInitializer: () -> TangentVector {
    { .zero }
  }
}

extension Empty: Vector {
  public var dimension: Int { return 0 }

  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {}

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {}

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {}

  @differentiable
  public func dot(_ other: Self) -> Double { return 0 }
}

extension Tuple: Vector
where Head: Vector, Tail: Vector {
  public var dimension: Int { return head.dimension + tail.dimension }

  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
    lhs.head += rhs.head
    lhs.tail += rhs.tail
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
    lhs.head -= rhs.head
    lhs.tail -= rhs.tail
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
    lhs.head *= rhs
    lhs.tail *= rhs
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    return head.dot(other.head) + tail.dot(other.tail)
  }
}

extension Empty: FixedSizeVector {
  public static var dimension: Int { 0 }
  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    assert(scalars.isEmpty)
    self.init()
  }
}

extension Tuple: FixedSizeVector, ScalarsInitializableVector
  where Head: FixedSizeVector, Tail: FixedSizeVector
{
  public static var dimension: Int { Head.dimension + Tail.dimension }
  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    let i = scalars.index(scalars.startIndex, offsetBy: Head.dimension)
    self.init(head: .init(scalars[..<i]), tail: .init(scalars[i...]))
  }
}
