// WARNING: This is a generated file. Do not edit it. Instead, edit the corresponding ".gyb" file.
// See "generate.sh" in the root of this repository for instructions how to regenerate files.

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 1)
import TensorFlow

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 10)

/// An element of R^1, with Euclidean inner product.
public struct Vector1: KeyPathIterable, TangentStandardBasis
{
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var x: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 17)

  @differentiable
  public init(_ x: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.x = x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 23)
  }
}

/// Conformance to EuclideanVector
extension Vector1: AdditiveArithmetic, EuclideanVector {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.x += rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 33)
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.x -= rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 40)
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.x *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 47)
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.x * other.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 55)
    return result
  }
}

/// Other arithmetic on the vector elements.
extension Vector1: ElementaryFunctions {
  /// Sum of the elements of `self`.
  @differentiable
  public func sum() -> Double {
    var result: Double = 0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 68)
    return result
  }

  /// Vector whose elements are squares of the elements of `self`.
  @differentiable
  public func squared() -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.x = x * x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 78)
    return result
  }
}

/// Conformance to `VectorConvertible`.
extension Vector1: VectorConvertible {
  @differentiable
  public init(_ vector: Vector) {
    var index = withoutDerivative(at: vector.scalars.startIndex)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.x = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 91)
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
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.x = tensor[0].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 121)
  }
}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 10)

/// An element of R^2, with Euclidean inner product.
public struct Vector2: KeyPathIterable, TangentStandardBasis
{
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var x: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var y: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 17)

  @differentiable
  public init(_ x: Double, _ y: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.x = x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.y = y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 23)
  }
}

/// Conformance to EuclideanVector
extension Vector2: AdditiveArithmetic, EuclideanVector {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.x += rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.y += rhs.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 33)
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.x -= rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.y -= rhs.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 40)
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.x *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.y *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 47)
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.x * other.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.y * other.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 55)
    return result
  }
}

/// Other arithmetic on the vector elements.
extension Vector2: ElementaryFunctions {
  /// Sum of the elements of `self`.
  @differentiable
  public func sum() -> Double {
    var result: Double = 0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 68)
    return result
  }

  /// Vector whose elements are squares of the elements of `self`.
  @differentiable
  public func squared() -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.x = x * x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.y = y * y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 78)
    return result
  }
}

/// Conformance to `VectorConvertible`.
extension Vector2: VectorConvertible {
  @differentiable
  public init(_ vector: Vector) {
    var index = withoutDerivative(at: vector.scalars.startIndex)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.x = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.y = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 91)
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
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.x = tensor[0].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.y = tensor[1].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 121)
  }
}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 10)

/// An element of R^3, with Euclidean inner product.
public struct Vector3: KeyPathIterable, TangentStandardBasis
{
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var x: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var y: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var z: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 17)

  @differentiable
  public init(_ x: Double, _ y: Double, _ z: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.x = x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.y = y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.z = z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 23)
  }
}

/// Conformance to EuclideanVector
extension Vector3: AdditiveArithmetic, EuclideanVector {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.x += rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.y += rhs.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.z += rhs.z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 33)
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.x -= rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.y -= rhs.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.z -= rhs.z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 40)
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.x *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.y *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.z *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 47)
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.x * other.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.y * other.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.z * other.z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 55)
    return result
  }
}

/// Other arithmetic on the vector elements.
extension Vector3: ElementaryFunctions {
  /// Sum of the elements of `self`.
  @differentiable
  public func sum() -> Double {
    var result: Double = 0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 68)
    return result
  }

  /// Vector whose elements are squares of the elements of `self`.
  @differentiable
  public func squared() -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.x = x * x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.y = y * y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.z = z * z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 78)
    return result
  }
}

/// Conformance to `VectorConvertible`.
extension Vector3: VectorConvertible {
  @differentiable
  public init(_ vector: Vector) {
    var index = withoutDerivative(at: vector.scalars.startIndex)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.x = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.y = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.z = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 91)
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
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.x = tensor[0].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.y = tensor[1].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.z = tensor[2].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 121)
  }
}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 10)

/// An element of R^4, with Euclidean inner product.
public struct Vector4: KeyPathIterable, TangentStandardBasis
{
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s0: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s1: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s2: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s3: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 17)

  @differentiable
  public init(_ s0: Double, _ s1: Double, _ s2: Double, _ s3: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s0 = s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s1 = s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s2 = s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s3 = s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 23)
  }
}

/// Conformance to EuclideanVector
extension Vector4: AdditiveArithmetic, EuclideanVector {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s0 += rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s1 += rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s2 += rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s3 += rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 33)
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s0 -= rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s1 -= rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s2 -= rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s3 -= rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 40)
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s0 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s1 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s2 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s3 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 47)
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s0 * other.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s1 * other.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s2 * other.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s3 * other.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 55)
    return result
  }
}

/// Other arithmetic on the vector elements.
extension Vector4: ElementaryFunctions {
  /// Sum of the elements of `self`.
  @differentiable
  public func sum() -> Double {
    var result: Double = 0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 68)
    return result
  }

  /// Vector whose elements are squares of the elements of `self`.
  @differentiable
  public func squared() -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s0 = s0 * s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s1 = s1 * s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s2 = s2 * s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s3 = s3 * s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 78)
    return result
  }
}

/// Conformance to `VectorConvertible`.
extension Vector4: VectorConvertible {
  @differentiable
  public init(_ vector: Vector) {
    var index = withoutDerivative(at: vector.scalars.startIndex)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s0 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s1 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s2 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s3 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 91)
  }

  @differentiable
  public var vector: Vector {
    return Vector([s0, s1, s2, s3])
  }
}

/// Conformance to `FixedDimensionVector`.
extension Vector4: FixedDimensionVector {
  public static var dimension: Int { return 4 }
}

/// Conversion to/from tensor.
extension Vector4 {
  /// A `Tensor` with shape `[4]` whose elements are the elements of `self`.
  @differentiable
  public var tensor: Tensor<Double> {
    Tensor([s0, s1, s2, s3])
  }

  /// Creates a `Vector4` with the same elements as `tensor`.
  ///
  /// Precondition: `tensor` must have shape `[4]`.
  @differentiable
  public init(_ tensor: Tensor<Double>) {
    precondition(tensor.shape == [4])
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s0 = tensor[0].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s1 = tensor[1].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s2 = tensor[2].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s3 = tensor[3].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 121)
  }
}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 10)

/// An element of R^5, with Euclidean inner product.
public struct Vector5: KeyPathIterable, TangentStandardBasis
{
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s0: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s1: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s2: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s3: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s4: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 17)

  @differentiable
  public init(_ s0: Double, _ s1: Double, _ s2: Double, _ s3: Double, _ s4: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s0 = s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s1 = s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s2 = s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s3 = s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s4 = s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 23)
  }
}

/// Conformance to EuclideanVector
extension Vector5: AdditiveArithmetic, EuclideanVector {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s0 += rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s1 += rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s2 += rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s3 += rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s4 += rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 33)
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s0 -= rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s1 -= rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s2 -= rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s3 -= rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s4 -= rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 40)
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s0 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s1 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s2 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s3 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s4 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 47)
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s0 * other.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s1 * other.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s2 * other.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s3 * other.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s4 * other.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 55)
    return result
  }
}

/// Other arithmetic on the vector elements.
extension Vector5: ElementaryFunctions {
  /// Sum of the elements of `self`.
  @differentiable
  public func sum() -> Double {
    var result: Double = 0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 68)
    return result
  }

  /// Vector whose elements are squares of the elements of `self`.
  @differentiable
  public func squared() -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s0 = s0 * s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s1 = s1 * s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s2 = s2 * s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s3 = s3 * s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s4 = s4 * s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 78)
    return result
  }
}

/// Conformance to `VectorConvertible`.
extension Vector5: VectorConvertible {
  @differentiable
  public init(_ vector: Vector) {
    var index = withoutDerivative(at: vector.scalars.startIndex)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s0 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s1 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s2 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s3 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s4 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 91)
  }

  @differentiable
  public var vector: Vector {
    return Vector([s0, s1, s2, s3, s4])
  }
}

/// Conformance to `FixedDimensionVector`.
extension Vector5: FixedDimensionVector {
  public static var dimension: Int { return 5 }
}

/// Conversion to/from tensor.
extension Vector5 {
  /// A `Tensor` with shape `[5]` whose elements are the elements of `self`.
  @differentiable
  public var tensor: Tensor<Double> {
    Tensor([s0, s1, s2, s3, s4])
  }

  /// Creates a `Vector5` with the same elements as `tensor`.
  ///
  /// Precondition: `tensor` must have shape `[5]`.
  @differentiable
  public init(_ tensor: Tensor<Double>) {
    precondition(tensor.shape == [5])
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s0 = tensor[0].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s1 = tensor[1].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s2 = tensor[2].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s3 = tensor[3].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s4 = tensor[4].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 121)
  }
}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 10)

/// An element of R^6, with Euclidean inner product.
public struct Vector6: KeyPathIterable, TangentStandardBasis
{
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s0: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s1: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s2: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s3: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s4: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s5: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 17)

  @differentiable
  public init(_ s0: Double, _ s1: Double, _ s2: Double, _ s3: Double, _ s4: Double, _ s5: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s0 = s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s1 = s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s2 = s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s3 = s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s4 = s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s5 = s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 23)
  }
}

/// Conformance to EuclideanVector
extension Vector6: AdditiveArithmetic, EuclideanVector {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s0 += rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s1 += rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s2 += rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s3 += rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s4 += rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s5 += rhs.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 33)
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s0 -= rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s1 -= rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s2 -= rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s3 -= rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s4 -= rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s5 -= rhs.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 40)
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s0 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s1 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s2 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s3 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s4 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s5 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 47)
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s0 * other.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s1 * other.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s2 * other.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s3 * other.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s4 * other.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s5 * other.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 55)
    return result
  }
}

/// Other arithmetic on the vector elements.
extension Vector6: ElementaryFunctions {
  /// Sum of the elements of `self`.
  @differentiable
  public func sum() -> Double {
    var result: Double = 0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 68)
    return result
  }

  /// Vector whose elements are squares of the elements of `self`.
  @differentiable
  public func squared() -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s0 = s0 * s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s1 = s1 * s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s2 = s2 * s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s3 = s3 * s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s4 = s4 * s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s5 = s5 * s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 78)
    return result
  }
}

/// Conformance to `VectorConvertible`.
extension Vector6: VectorConvertible {
  @differentiable
  public init(_ vector: Vector) {
    var index = withoutDerivative(at: vector.scalars.startIndex)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s0 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s1 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s2 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s3 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s4 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s5 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 91)
  }

  @differentiable
  public var vector: Vector {
    return Vector([s0, s1, s2, s3, s4, s5])
  }
}

/// Conformance to `FixedDimensionVector`.
extension Vector6: FixedDimensionVector {
  public static var dimension: Int { return 6 }
}

/// Conversion to/from tensor.
extension Vector6 {
  /// A `Tensor` with shape `[6]` whose elements are the elements of `self`.
  @differentiable
  public var tensor: Tensor<Double> {
    Tensor([s0, s1, s2, s3, s4, s5])
  }

  /// Creates a `Vector6` with the same elements as `tensor`.
  ///
  /// Precondition: `tensor` must have shape `[6]`.
  @differentiable
  public init(_ tensor: Tensor<Double>) {
    precondition(tensor.shape == [6])
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s0 = tensor[0].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s1 = tensor[1].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s2 = tensor[2].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s3 = tensor[3].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s4 = tensor[4].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s5 = tensor[5].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 121)
  }
}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 10)

/// An element of R^7, with Euclidean inner product.
public struct Vector7: KeyPathIterable, TangentStandardBasis
{
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s0: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s1: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s2: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s3: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s4: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s5: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s6: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 17)

  @differentiable
  public init(_ s0: Double, _ s1: Double, _ s2: Double, _ s3: Double, _ s4: Double, _ s5: Double, _ s6: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s0 = s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s1 = s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s2 = s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s3 = s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s4 = s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s5 = s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s6 = s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 23)
  }
}

/// Conformance to EuclideanVector
extension Vector7: AdditiveArithmetic, EuclideanVector {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s0 += rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s1 += rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s2 += rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s3 += rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s4 += rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s5 += rhs.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s6 += rhs.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 33)
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s0 -= rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s1 -= rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s2 -= rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s3 -= rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s4 -= rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s5 -= rhs.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s6 -= rhs.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 40)
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s0 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s1 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s2 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s3 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s4 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s5 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s6 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 47)
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s0 * other.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s1 * other.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s2 * other.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s3 * other.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s4 * other.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s5 * other.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s6 * other.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 55)
    return result
  }
}

/// Other arithmetic on the vector elements.
extension Vector7: ElementaryFunctions {
  /// Sum of the elements of `self`.
  @differentiable
  public func sum() -> Double {
    var result: Double = 0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 68)
    return result
  }

  /// Vector whose elements are squares of the elements of `self`.
  @differentiable
  public func squared() -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s0 = s0 * s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s1 = s1 * s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s2 = s2 * s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s3 = s3 * s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s4 = s4 * s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s5 = s5 * s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s6 = s6 * s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 78)
    return result
  }
}

/// Conformance to `VectorConvertible`.
extension Vector7: VectorConvertible {
  @differentiable
  public init(_ vector: Vector) {
    var index = withoutDerivative(at: vector.scalars.startIndex)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s0 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s1 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s2 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s3 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s4 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s5 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s6 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 91)
  }

  @differentiable
  public var vector: Vector {
    return Vector([s0, s1, s2, s3, s4, s5, s6])
  }
}

/// Conformance to `FixedDimensionVector`.
extension Vector7: FixedDimensionVector {
  public static var dimension: Int { return 7 }
}

/// Conversion to/from tensor.
extension Vector7 {
  /// A `Tensor` with shape `[7]` whose elements are the elements of `self`.
  @differentiable
  public var tensor: Tensor<Double> {
    Tensor([s0, s1, s2, s3, s4, s5, s6])
  }

  /// Creates a `Vector7` with the same elements as `tensor`.
  ///
  /// Precondition: `tensor` must have shape `[7]`.
  @differentiable
  public init(_ tensor: Tensor<Double>) {
    precondition(tensor.shape == [7])
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s0 = tensor[0].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s1 = tensor[1].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s2 = tensor[2].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s3 = tensor[3].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s4 = tensor[4].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s5 = tensor[5].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s6 = tensor[6].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 121)
  }
}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 10)

/// An element of R^8, with Euclidean inner product.
public struct Vector8: KeyPathIterable, TangentStandardBasis
{
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s0: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s1: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s2: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s3: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s4: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s5: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s6: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s7: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 17)

  @differentiable
  public init(_ s0: Double, _ s1: Double, _ s2: Double, _ s3: Double, _ s4: Double, _ s5: Double, _ s6: Double, _ s7: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s0 = s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s1 = s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s2 = s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s3 = s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s4 = s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s5 = s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s6 = s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s7 = s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 23)
  }
}

/// Conformance to EuclideanVector
extension Vector8: AdditiveArithmetic, EuclideanVector {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s0 += rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s1 += rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s2 += rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s3 += rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s4 += rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s5 += rhs.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s6 += rhs.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s7 += rhs.s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 33)
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s0 -= rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s1 -= rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s2 -= rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s3 -= rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s4 -= rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s5 -= rhs.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s6 -= rhs.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s7 -= rhs.s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 40)
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s0 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s1 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s2 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s3 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s4 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s5 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s6 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s7 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 47)
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s0 * other.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s1 * other.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s2 * other.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s3 * other.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s4 * other.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s5 * other.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s6 * other.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s7 * other.s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 55)
    return result
  }
}

/// Other arithmetic on the vector elements.
extension Vector8: ElementaryFunctions {
  /// Sum of the elements of `self`.
  @differentiable
  public func sum() -> Double {
    var result: Double = 0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 68)
    return result
  }

  /// Vector whose elements are squares of the elements of `self`.
  @differentiable
  public func squared() -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s0 = s0 * s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s1 = s1 * s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s2 = s2 * s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s3 = s3 * s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s4 = s4 * s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s5 = s5 * s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s6 = s6 * s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s7 = s7 * s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 78)
    return result
  }
}

/// Conformance to `VectorConvertible`.
extension Vector8: VectorConvertible {
  @differentiable
  public init(_ vector: Vector) {
    var index = withoutDerivative(at: vector.scalars.startIndex)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s0 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s1 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s2 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s3 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s4 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s5 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s6 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s7 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 91)
  }

  @differentiable
  public var vector: Vector {
    return Vector([s0, s1, s2, s3, s4, s5, s6, s7])
  }
}

/// Conformance to `FixedDimensionVector`.
extension Vector8: FixedDimensionVector {
  public static var dimension: Int { return 8 }
}

/// Conversion to/from tensor.
extension Vector8 {
  /// A `Tensor` with shape `[8]` whose elements are the elements of `self`.
  @differentiable
  public var tensor: Tensor<Double> {
    Tensor([s0, s1, s2, s3, s4, s5, s6, s7])
  }

  /// Creates a `Vector8` with the same elements as `tensor`.
  ///
  /// Precondition: `tensor` must have shape `[8]`.
  @differentiable
  public init(_ tensor: Tensor<Double>) {
    precondition(tensor.shape == [8])
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s0 = tensor[0].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s1 = tensor[1].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s2 = tensor[2].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s3 = tensor[3].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s4 = tensor[4].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s5 = tensor[5].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s6 = tensor[6].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s7 = tensor[7].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 121)
  }
}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 10)

/// An element of R^9, with Euclidean inner product.
public struct Vector9: KeyPathIterable, TangentStandardBasis
{
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s0: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s1: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s2: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s3: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s4: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s5: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s6: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s7: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 15)
  @differentiable public var s8: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 17)

  @differentiable
  public init(_ s0: Double, _ s1: Double, _ s2: Double, _ s3: Double, _ s4: Double, _ s5: Double, _ s6: Double, _ s7: Double, _ s8: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s0 = s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s1 = s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s2 = s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s3 = s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s4 = s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s5 = s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s6 = s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s7 = s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 21)
    self.s8 = s8
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 23)
  }
}

/// Conformance to EuclideanVector
extension Vector9: AdditiveArithmetic, EuclideanVector {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s0 += rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s1 += rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s2 += rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s3 += rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s4 += rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s5 += rhs.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s6 += rhs.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s7 += rhs.s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 31)
    lhs.s8 += rhs.s8
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 33)
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s0 -= rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s1 -= rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s2 -= rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s3 -= rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s4 -= rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s5 -= rhs.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s6 -= rhs.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s7 -= rhs.s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 38)
    lhs.s8 -= rhs.s8
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 40)
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s0 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s1 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s2 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s3 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s4 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s5 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s6 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s7 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 45)
    lhs.s8 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 47)
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s0 * other.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s1 * other.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s2 * other.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s3 * other.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s4 * other.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s5 * other.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s6 * other.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s7 * other.s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 53)
    result += self.s8 * other.s8
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 55)
    return result
  }
}

/// Other arithmetic on the vector elements.
extension Vector9: ElementaryFunctions {
  /// Sum of the elements of `self`.
  @differentiable
  public func sum() -> Double {
    var result: Double = 0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 66)
    result = result + s8
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 68)
    return result
  }

  /// Vector whose elements are squares of the elements of `self`.
  @differentiable
  public func squared() -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s0 = s0 * s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s1 = s1 * s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s2 = s2 * s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s3 = s3 * s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s4 = s4 * s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s5 = s5 * s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s6 = s6 * s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s7 = s7 * s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 76)
    result.s8 = s8 * s8
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 78)
    return result
  }
}

/// Conformance to `VectorConvertible`.
extension Vector9: VectorConvertible {
  @differentiable
  public init(_ vector: Vector) {
    var index = withoutDerivative(at: vector.scalars.startIndex)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s0 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s1 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s2 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s3 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s4 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s5 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s6 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s7 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 88)
    self.s8 = vector.scalars[index]
    index = withoutDerivative(at: vector.scalars.index(after: index))
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 91)
  }

  @differentiable
  public var vector: Vector {
    return Vector([s0, s1, s2, s3, s4, s5, s6, s7, s8])
  }
}

/// Conformance to `FixedDimensionVector`.
extension Vector9: FixedDimensionVector {
  public static var dimension: Int { return 9 }
}

/// Conversion to/from tensor.
extension Vector9 {
  /// A `Tensor` with shape `[9]` whose elements are the elements of `self`.
  @differentiable
  public var tensor: Tensor<Double> {
    Tensor([s0, s1, s2, s3, s4, s5, s6, s7, s8])
  }

  /// Creates a `Vector9` with the same elements as `tensor`.
  ///
  /// Precondition: `tensor` must have shape `[9]`.
  @differentiable
  public init(_ tensor: Tensor<Double>) {
    precondition(tensor.shape == [9])
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s0 = tensor[0].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s1 = tensor[1].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s2 = tensor[2].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s3 = tensor[3].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s4 = tensor[4].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s5 = tensor[5].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s6 = tensor[6].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s7 = tensor[7].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 119)
    self.s8 = tensor[8].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 121)
  }
}

