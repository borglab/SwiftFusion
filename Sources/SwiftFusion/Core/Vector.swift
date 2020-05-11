// WARNING: This is a generated file. Do not edit it. Instead, edit the corresponding ".gyb" file.
// See "generate.sh" in the root of this repository for instructions how to regenerate files.

// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 1)
import TensorFlow

// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 6)

/// An element of R^1, with Euclidean norm.
public struct Vector1: KeyPathIterable, TangentStandardBasis
{
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 11)
  @differentiable public var x: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 13)

  @differentiable
  public init(_ x: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 17)
    self.x = x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 19)
  }
}

/// EuclideanVectorSpace conformance.
extension Vector1: EuclideanVectorSpace {
  public typealias VectorSpaceScalar = Double
  public typealias TangentVector = Self

  /// Euclidean norm of `self`.
  @differentiable
  public var norm: Double { squaredNorm.squareRoot() }

  /// Square of the Euclidean norm of `self`.
  @differentiable
  public var squaredNorm: Double { self.squared().sum() }

  @differentiable
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 39)
    result.x = lhs.x + rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 41)
    return result
  }

  @differentiable
  public static func - (_ lhs: Self, _ rhs: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 48)
    result.x = lhs.x - rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 50)
    return result
  }

  @differentiable
  public static prefix func - (_ v: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 57)
    result.x = -v.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 59)
    return result
  }
}

/// Other arithmetic on the vector elements.
extension Vector1: ElementaryFunctions {
  /// Sum of the elements of `self`.
  public func sum() -> Double {
    var result: Double = 0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 69)
    result = result + x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 71)
    return result
  }

  /// Vector whose elements are squares of the elements of `self`.
  public func squared() -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 78)
    result.x = x * x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 80)
    return result
  }
}

/// Conversion to/from tensor.
extension Vector1 {
  /// A `Tensor` with shape `[1]` whose elements are the elements of `self`.
  @differentiable
  public var tensor: Tensor<Double> {
    Tensor([x])
  }

  /// Creates a `Vector1` with the same elements as `tensor`.
  ///
  /// Precondition: `tensor` must have shape `[1]`.
  @differentiable
  public init(_ tensor: Tensor<Double>) {
    precondition(tensor.shape == [1])
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 99)
    self.x = tensor[0].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 101)
  }
}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 6)

/// An element of R^2, with Euclidean norm.
public struct Vector2: KeyPathIterable, TangentStandardBasis
{
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 11)
  @differentiable public var x: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 11)
  @differentiable public var y: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 13)

  @differentiable
  public init(_ x: Double, _ y: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 17)
    self.x = x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 17)
    self.y = y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 19)
  }
}

/// EuclideanVectorSpace conformance.
extension Vector2: EuclideanVectorSpace {
  public typealias VectorSpaceScalar = Double
  public typealias TangentVector = Self

  /// Euclidean norm of `self`.
  @differentiable
  public var norm: Double { squaredNorm.squareRoot() }

  /// Square of the Euclidean norm of `self`.
  @differentiable
  public var squaredNorm: Double { self.squared().sum() }

  @differentiable
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 39)
    result.x = lhs.x + rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 39)
    result.y = lhs.y + rhs.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 41)
    return result
  }

  @differentiable
  public static func - (_ lhs: Self, _ rhs: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 48)
    result.x = lhs.x - rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 48)
    result.y = lhs.y - rhs.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 50)
    return result
  }

  @differentiable
  public static prefix func - (_ v: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 57)
    result.x = -v.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 57)
    result.y = -v.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 59)
    return result
  }
}

/// Other arithmetic on the vector elements.
extension Vector2: ElementaryFunctions {
  /// Sum of the elements of `self`.
  public func sum() -> Double {
    var result: Double = 0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 69)
    result = result + x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 69)
    result = result + y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 71)
    return result
  }

  /// Vector whose elements are squares of the elements of `self`.
  public func squared() -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 78)
    result.x = x * x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 78)
    result.y = y * y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 80)
    return result
  }
}

/// Conversion to/from tensor.
extension Vector2 {
  /// A `Tensor` with shape `[2]` whose elements are the elements of `self`.
  @differentiable
  public var tensor: Tensor<Double> {
    Tensor([x, y])
  }

  /// Creates a `Vector2` with the same elements as `tensor`.
  ///
  /// Precondition: `tensor` must have shape `[2]`.
  @differentiable
  public init(_ tensor: Tensor<Double>) {
    precondition(tensor.shape == [2])
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 99)
    self.x = tensor[0].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 99)
    self.y = tensor[1].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 101)
  }
}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 6)

/// An element of R^3, with Euclidean norm.
public struct Vector3: KeyPathIterable, TangentStandardBasis
{
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 11)
  @differentiable public var x: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 11)
  @differentiable public var y: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 11)
  @differentiable public var z: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 13)

  @differentiable
  public init(_ x: Double, _ y: Double, _ z: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 17)
    self.x = x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 17)
    self.y = y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 17)
    self.z = z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 19)
  }
}

/// EuclideanVectorSpace conformance.
extension Vector3: EuclideanVectorSpace {
  public typealias VectorSpaceScalar = Double
  public typealias TangentVector = Self

  /// Euclidean norm of `self`.
  @differentiable
  public var norm: Double { squaredNorm.squareRoot() }

  /// Square of the Euclidean norm of `self`.
  @differentiable
  public var squaredNorm: Double { self.squared().sum() }

  @differentiable
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 39)
    result.x = lhs.x + rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 39)
    result.y = lhs.y + rhs.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 39)
    result.z = lhs.z + rhs.z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 41)
    return result
  }

  @differentiable
  public static func - (_ lhs: Self, _ rhs: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 48)
    result.x = lhs.x - rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 48)
    result.y = lhs.y - rhs.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 48)
    result.z = lhs.z - rhs.z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 50)
    return result
  }

  @differentiable
  public static prefix func - (_ v: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 57)
    result.x = -v.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 57)
    result.y = -v.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 57)
    result.z = -v.z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 59)
    return result
  }
}

/// Other arithmetic on the vector elements.
extension Vector3: ElementaryFunctions {
  /// Sum of the elements of `self`.
  public func sum() -> Double {
    var result: Double = 0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 69)
    result = result + x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 69)
    result = result + y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 69)
    result = result + z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 71)
    return result
  }

  /// Vector whose elements are squares of the elements of `self`.
  public func squared() -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 78)
    result.x = x * x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 78)
    result.y = y * y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 78)
    result.z = z * z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 80)
    return result
  }
}

/// Conversion to/from tensor.
extension Vector3 {
  /// A `Tensor` with shape `[3]` whose elements are the elements of `self`.
  @differentiable
  public var tensor: Tensor<Double> {
    Tensor([x, y, z])
  }

  /// Creates a `Vector3` with the same elements as `tensor`.
  ///
  /// Precondition: `tensor` must have shape `[3]`.
  @differentiable
  public init(_ tensor: Tensor<Double>) {
    precondition(tensor.shape == [3])
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 99)
    self.x = tensor[0].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 99)
    self.y = tensor[1].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 99)
    self.z = tensor[2].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 101)
  }
}

