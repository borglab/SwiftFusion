// WARNING: This is a generated file. Do not edit it. Instead, edit the corresponding ".gyb" file.
// See "generate.sh" in the root of this repository for instructions how to regenerate files.

// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 1)
import TensorFlow

// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 6)

/// An element of R^1.
public struct Vector1:
  AdditiveArithmetic, Differentiable, ElementaryFunctions, KeyPathIterable, TangentStandardBasis
{
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 12)
  @differentiable public var x: Double
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 14)

  @differentiable
  public init(_ x: Double) {
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 18)
    self.x = x
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 20)
  }
}

/// Normed vector space methods.
extension Vector1: VectorProtocol {
  public typealias VectorSpaceScalar = Double

  @differentiable
  public var magnitude: Double {
    var squaredMagnitude: Double = 0
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 31)
    squaredMagnitude = squaredMagnitude + x * x
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 33)
    return squaredMagnitude.squareRoot()
  }

  @differentiable
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 40)
    result.x = lhs.x + rhs.x
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 42)
    return result
  }

  @differentiable
  public static func - (_ lhs: Self, _ rhs: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 49)
    result.x = lhs.x - rhs.x
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 51)
    return result
  }

  @differentiable
  public static prefix func - (_ v: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 58)
    result.x = -v.x
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 60)
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
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 79)
    self.x = tensor[0].scalarized()
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 81)
  }
}

// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 6)

/// An element of R^2.
public struct Vector2:
  AdditiveArithmetic, Differentiable, ElementaryFunctions, KeyPathIterable, TangentStandardBasis
{
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 12)
  @differentiable public var x: Double
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 12)
  @differentiable public var y: Double
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 14)

  @differentiable
  public init(_ x: Double, _ y: Double) {
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 18)
    self.x = x
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 18)
    self.y = y
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 20)
  }
}

/// Normed vector space methods.
extension Vector2: VectorProtocol {
  public typealias VectorSpaceScalar = Double

  @differentiable
  public var magnitude: Double {
    var squaredMagnitude: Double = 0
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 31)
    squaredMagnitude = squaredMagnitude + x * x
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 31)
    squaredMagnitude = squaredMagnitude + y * y
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 33)
    return squaredMagnitude.squareRoot()
  }

  @differentiable
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 40)
    result.x = lhs.x + rhs.x
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 40)
    result.y = lhs.y + rhs.y
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 42)
    return result
  }

  @differentiable
  public static func - (_ lhs: Self, _ rhs: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 49)
    result.x = lhs.x - rhs.x
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 49)
    result.y = lhs.y - rhs.y
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 51)
    return result
  }

  @differentiable
  public static prefix func - (_ v: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 58)
    result.x = -v.x
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 58)
    result.y = -v.y
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 60)
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
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 79)
    self.x = tensor[0].scalarized()
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 79)
    self.y = tensor[1].scalarized()
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 81)
  }
}

// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 6)

/// An element of R^3.
public struct Vector3:
  AdditiveArithmetic, Differentiable, ElementaryFunctions, KeyPathIterable, TangentStandardBasis
{
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 12)
  @differentiable public var x: Double
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 12)
  @differentiable public var y: Double
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 12)
  @differentiable public var z: Double
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 14)

  @differentiable
  public init(_ x: Double, _ y: Double, _ z: Double) {
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 18)
    self.x = x
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 18)
    self.y = y
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 18)
    self.z = z
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 20)
  }
}

/// Normed vector space methods.
extension Vector3: VectorProtocol {
  public typealias VectorSpaceScalar = Double

  @differentiable
  public var magnitude: Double {
    var squaredMagnitude: Double = 0
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 31)
    squaredMagnitude = squaredMagnitude + x * x
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 31)
    squaredMagnitude = squaredMagnitude + y * y
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 31)
    squaredMagnitude = squaredMagnitude + z * z
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 33)
    return squaredMagnitude.squareRoot()
  }

  @differentiable
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 40)
    result.x = lhs.x + rhs.x
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 40)
    result.y = lhs.y + rhs.y
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 40)
    result.z = lhs.z + rhs.z
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 42)
    return result
  }

  @differentiable
  public static func - (_ lhs: Self, _ rhs: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 49)
    result.x = lhs.x - rhs.x
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 49)
    result.y = lhs.y - rhs.y
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 49)
    result.z = lhs.z - rhs.z
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 51)
    return result
  }

  @differentiable
  public static prefix func - (_ v: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 58)
    result.x = -v.x
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 58)
    result.y = -v.y
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 58)
    result.z = -v.z
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 60)
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
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 79)
    self.x = tensor[0].scalarized()
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 79)
    self.y = tensor[1].scalarized()
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 79)
    self.z = tensor[2].scalarized()
// ###sourceLocation(file: "/usr/local/google/home/marcrasi/git/SwiftFusion/Sources/SwiftFusion/Core/Vector.swift.gyb", line: 81)
  }
}

