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
}

extension Tuple: Differentiable
where Head: Differentiable, Tail: Differentiable, Tail.TangentVector: TupleProtocol {
  public typealias TangentVector = Tuple<Head.TangentVector, Tail.TangentVector>
  public mutating func move(along direction: TangentVector) {
    head.move(along: direction.head)
    tail.move(along: direction.tail)
  }
}

extension Empty: EuclideanVector {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {}

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {}

  // MARK: - Scalar multiplication.

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {}

  // MARK: - Euclidean structure.

  /// The inner product of `self` with `other`.
  @differentiable
  public func dot(_ other: Self) -> Double { return 0 }

  // MARK: - Conversion to/from collections of scalars.

  /// Creates an instance whose elements are `scalars`.
  ///
  /// Precondition: `scalars` must be empty.
  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    assert(scalars.isEmpty)
    self.init()
  }

  /// The scalars in `self`.
  public var scalars: [Double] { [] }
}

extension Empty: EuclideanVectorN {
  /// The dimension of the vector.
  public static var dimension: Int { return 0 }

  /// A standard basis of vectors.
  public static var standardBasis: [Self] { return [] }
}

extension Tuple: EuclideanVector, EuclideanVectorN
where Head: EuclideanVectorN, Tail: EuclideanVectorN {
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

  // MARK: - Scalar multiplication.

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
    lhs.head *= rhs
    lhs.tail *= rhs
  }

  // MARK: - Euclidean structure.

  /// The inner product of `self` with `other`.
  @differentiable
  public func dot(_ other: Self) -> Double {
    return head.dot(other.head) + tail.dot(other.tail)
  }

  // MARK: - Conversion to/from collections of scalars.

  /// Creates an instance whose elements are `scalars`.
  ///
  /// The first `Head.dimension` scalars go in `self.head`, and the remaining `Tail.dimension`
  /// scalars go in `self.tail`.
  ///
  /// Precondition: `scalars` must have exactly `Head.dimension + Tail.dimension` elements.
  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    self.init(
      head: Head(scalars.prefix(Head.dimension)),
      tail: Tail(scalars.dropFirst(Head.dimension))
    )
  }

  /// The scalars in `self`.
  public var scalars: [Double] {
    // Note: Not spending effort making this more efficient because we're going to stop using this
    // soon.
    return Array(head.scalars) + Array(tail.scalars)
  }

  // MARK: `EuclideanVectorN` requirements.

  /// The dimension of the vector.
  public static var dimension: Int { return Head.dimension + Tail.dimension }

  /// A standard basis of vectors.
  public static var standardBasis: [Self] {
    return Head.standardBasis.map { Self(head: $0, tail: Tail.zero) }
      + Tail.standardBasis.map { Self(head: Head.zero, tail: $0) }
  }
}
