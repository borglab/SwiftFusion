// WARNING: This is a generated file. Do not edit it. Instead, edit the corresponding ".gyb" file.
// See "generate.sh" in the root of this repository for instructions how to regenerate files.

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 1)
import TensorFlow

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 6)

/// An element of R^1, with Euclidean norm.
public struct Vector1: EuclideanVectorSpace, KeyPathIterable, TangentStandardBasis
{
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 11)
  @differentiable public var x: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 13)

  @differentiable
  public init(_ x: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 17)
    self.x = x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 19)
  }
}

/// Normed vector space methods.
extension Vector1: AdditiveArithmetic, VectorProtocol {
  public typealias VectorSpaceScalar = Double

  /// Euclidean norm of `self`.
  @differentiable
  public var norm: Double { squaredNorm.squareRoot() }

  /// Square of the Euclidean norm of `self`.
  @differentiable
  public var squaredNorm: Double { self.squared().sum() }

  @differentiable
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    result.x = lhs.x + rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 40)
    return result
  }

  @differentiable
  public static func - (_ lhs: Self, _ rhs: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 47)
    result.x = lhs.x - rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 49)
    return result
  }

  @differentiable
  public static prefix func - (_ v: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 56)
    result.x = -v.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 58)
    return result
  }
}

/// Other arithmetic on the vector elements.
extension Vector1: ElementaryFunctions {
  /// Sum of the elements of `self`.
  public func sum() -> Double {
    var result: Double = 0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 68)
    result = result + x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    return result
  }

  /// Vector whose elements are squares of the elements of `self`.
  public func squared() -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 77)
    result.x = x * x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 79)
    return result
  }
}

/// Conformance to `VectorConvertible`.
extension Vector1: VectorConvertible {
  @differentiable
  public init(_ vector: Vector) {
    var index = withoutDerivative(at: vector.scalars.startIndex)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 89)
    self.x = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 92)
  }

  @differentiable
  public var vector: Vector {
    return Vector([x])
  }
}

/// Conformance to `FixedDimensionVector`.
extension Vector1: FixedDimensionVector {
  public static var dimension: Int { return 1 }
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
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 120)
    self.x = tensor[0].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 122)
  }
}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 6)

/// An element of R^2, with Euclidean norm.
public struct Vector2: EuclideanVectorSpace, KeyPathIterable, TangentStandardBasis
{
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 11)
  @differentiable public var x: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 11)
  @differentiable public var y: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 13)

  @differentiable
  public init(_ x: Double, _ y: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 17)
    self.x = x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 17)
    self.y = y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 19)
  }
}

/// Normed vector space methods.
extension Vector2: AdditiveArithmetic, VectorProtocol {
  public typealias VectorSpaceScalar = Double

  /// Euclidean norm of `self`.
  @differentiable
  public var norm: Double { squaredNorm.squareRoot() }

  /// Square of the Euclidean norm of `self`.
  @differentiable
  public var squaredNorm: Double { self.squared().sum() }

  @differentiable
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    result.x = lhs.x + rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    result.y = lhs.y + rhs.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 40)
    return result
  }

  @differentiable
  public static func - (_ lhs: Self, _ rhs: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 47)
    result.x = lhs.x - rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 47)
    result.y = lhs.y - rhs.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 49)
    return result
  }

  @differentiable
  public static prefix func - (_ v: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 56)
    result.x = -v.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 56)
    result.y = -v.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 58)
    return result
  }
}

/// Other arithmetic on the vector elements.
extension Vector2: ElementaryFunctions {
  /// Sum of the elements of `self`.
  public func sum() -> Double {
    var result: Double = 0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 68)
    result = result + x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 68)
    result = result + y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    return result
  }

  /// Vector whose elements are squares of the elements of `self`.
  public func squared() -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 77)
    result.x = x * x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 77)
    result.y = y * y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 79)
    return result
  }
}

/// Conformance to `VectorConvertible`.
extension Vector2: VectorConvertible {
  @differentiable
  public init(_ vector: Vector) {
    var index = withoutDerivative(at: vector.scalars.startIndex)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 89)
    self.x = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 89)
    self.y = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 92)
  }

  @differentiable
  public var vector: Vector {
    return Vector([x, y])
  }
}

/// Conformance to `FixedDimensionVector`.
extension Vector2: FixedDimensionVector {
  public static var dimension: Int { return 2 }
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
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 120)
    self.x = tensor[0].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 120)
    self.y = tensor[1].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 122)
  }
}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 6)

/// An element of R^3, with Euclidean norm.
public struct Vector3: EuclideanVectorSpace, KeyPathIterable, TangentStandardBasis
{
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 11)
  @differentiable public var x: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 11)
  @differentiable public var y: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 11)
  @differentiable public var z: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 13)

  @differentiable
  public init(_ x: Double, _ y: Double, _ z: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 17)
    self.x = x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 17)
    self.y = y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 17)
    self.z = z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 19)
  }
}

/// Normed vector space methods.
extension Vector3: AdditiveArithmetic, VectorProtocol {
  public typealias VectorSpaceScalar = Double

  /// Euclidean norm of `self`.
  @differentiable
  public var norm: Double { squaredNorm.squareRoot() }

  /// Square of the Euclidean norm of `self`.
  @differentiable
  public var squaredNorm: Double { self.squared().sum() }

  @differentiable
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    result.x = lhs.x + rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    result.y = lhs.y + rhs.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    result.z = lhs.z + rhs.z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 40)
    return result
  }

  @differentiable
  public static func - (_ lhs: Self, _ rhs: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 47)
    result.x = lhs.x - rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 47)
    result.y = lhs.y - rhs.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 47)
    result.z = lhs.z - rhs.z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 49)
    return result
  }

  @differentiable
  public static prefix func - (_ v: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 56)
    result.x = -v.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 56)
    result.y = -v.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 56)
    result.z = -v.z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 58)
    return result
  }
}

/// Other arithmetic on the vector elements.
extension Vector3: ElementaryFunctions {
  /// Sum of the elements of `self`.
  public func sum() -> Double {
    var result: Double = 0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 68)
    result = result + x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 68)
    result = result + y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 68)
    result = result + z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    return result
  }

  /// Vector whose elements are squares of the elements of `self`.
  public func squared() -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 77)
    result.x = x * x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 77)
    result.y = y * y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 77)
    result.z = z * z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 79)
    return result
  }
}

/// Conformance to `VectorConvertible`.
extension Vector3: VectorConvertible {
  @differentiable
  public init(_ vector: Vector) {
    var index = withoutDerivative(at: vector.scalars.startIndex)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 89)
    self.x = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 89)
    self.y = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 89)
    self.z = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 92)
  }

  @differentiable
  public var vector: Vector {
    return Vector([x, y, z])
  }
}

/// Conformance to `FixedDimensionVector`.
extension Vector3: FixedDimensionVector {
  public static var dimension: Int { return 3 }
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
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 120)
    self.x = tensor[0].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 120)
    self.y = tensor[1].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 120)
    self.z = tensor[2].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 122)
  }
}

