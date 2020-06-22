// WARNING: This is a generated file. Do not edit it. Instead, edit the corresponding ".gyb" file.
// See "generate.sh" in the root of this repository for instructions how to regenerate files.

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 1)
import TensorFlow

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 10)

/// An element of R^1, with Euclidean inner product.
public struct Vector1: KeyPathIterable {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var x: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 16)

  @differentiable
  public init(_ x: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.x = x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 22)
  }
}

/// Conformance to EuclideanVectorN
extension Vector1: AdditiveArithmetic, EuclideanVectorN {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.x += rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 32)
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.x -= rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 39)
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.x *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 46)
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.x * other.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 54)
    return result
  }

  public static var dimension: Int { return 1 }

  public static var standardBasis: [Self] {
    var result = Array(repeating: Self.zero, count: 1)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[0].x = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 64)
    return result
  }

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    var index = scalars.startIndex
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.x = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 73)
  }

  public var scalars: [Double] {
    return [x]
  }
}

extension Vector1: ElementaryFunctions {}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 10)

/// An element of R^2, with Euclidean inner product.
public struct Vector2: KeyPathIterable {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var x: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var y: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 16)

  @differentiable
  public init(_ x: Double, _ y: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.x = x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.y = y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 22)
  }
}

/// Conformance to EuclideanVectorN
extension Vector2: AdditiveArithmetic, EuclideanVectorN {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.x += rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.y += rhs.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 32)
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.x -= rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.y -= rhs.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 39)
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.x *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.y *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 46)
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.x * other.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.y * other.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 54)
    return result
  }

  public static var dimension: Int { return 2 }

  public static var standardBasis: [Self] {
    var result = Array(repeating: Self.zero, count: 2)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[0].x = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[1].y = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 64)
    return result
  }

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    var index = scalars.startIndex
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.x = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.y = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 73)
  }

  public var scalars: [Double] {
    return [x, y]
  }
}

extension Vector2: ElementaryFunctions {}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 10)

/// An element of R^3, with Euclidean inner product.
public struct Vector3: KeyPathIterable {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var x: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var y: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var z: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 16)

  @differentiable
  public init(_ x: Double, _ y: Double, _ z: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.x = x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.y = y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.z = z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 22)
  }
}

/// Conformance to EuclideanVectorN
extension Vector3: AdditiveArithmetic, EuclideanVectorN {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.x += rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.y += rhs.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.z += rhs.z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 32)
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.x -= rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.y -= rhs.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.z -= rhs.z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 39)
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.x *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.y *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.z *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 46)
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.x * other.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.y * other.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.z * other.z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 54)
    return result
  }

  public static var dimension: Int { return 3 }

  public static var standardBasis: [Self] {
    var result = Array(repeating: Self.zero, count: 3)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[0].x = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[1].y = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[2].z = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 64)
    return result
  }

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    var index = scalars.startIndex
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.x = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.y = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.z = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 73)
  }

  public var scalars: [Double] {
    return [x, y, z]
  }
}

extension Vector3: ElementaryFunctions {}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 10)

/// An element of R^4, with Euclidean inner product.
public struct Vector4: KeyPathIterable {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s0: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s1: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s2: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s3: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 16)

  @differentiable
  public init(_ s0: Double, _ s1: Double, _ s2: Double, _ s3: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s0 = s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s1 = s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s2 = s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s3 = s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 22)
  }
}

/// Conformance to EuclideanVectorN
extension Vector4: AdditiveArithmetic, EuclideanVectorN {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s0 += rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s1 += rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s2 += rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s3 += rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 32)
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s0 -= rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s1 -= rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s2 -= rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s3 -= rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 39)
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s0 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s1 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s2 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s3 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 46)
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s0 * other.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s1 * other.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s2 * other.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s3 * other.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 54)
    return result
  }

  public static var dimension: Int { return 4 }

  public static var standardBasis: [Self] {
    var result = Array(repeating: Self.zero, count: 4)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[0].s0 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[1].s1 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[2].s2 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[3].s3 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 64)
    return result
  }

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    var index = scalars.startIndex
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s0 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s1 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s2 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s3 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 73)
  }

  public var scalars: [Double] {
    return [s0, s1, s2, s3]
  }
}

extension Vector4: ElementaryFunctions {}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 10)

/// An element of R^5, with Euclidean inner product.
public struct Vector5: KeyPathIterable {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s0: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s1: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s2: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s3: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s4: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 16)

  @differentiable
  public init(_ s0: Double, _ s1: Double, _ s2: Double, _ s3: Double, _ s4: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s0 = s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s1 = s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s2 = s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s3 = s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s4 = s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 22)
  }
}

/// Conformance to EuclideanVectorN
extension Vector5: AdditiveArithmetic, EuclideanVectorN {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s0 += rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s1 += rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s2 += rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s3 += rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s4 += rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 32)
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s0 -= rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s1 -= rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s2 -= rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s3 -= rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s4 -= rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 39)
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s0 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s1 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s2 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s3 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s4 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 46)
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s0 * other.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s1 * other.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s2 * other.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s3 * other.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s4 * other.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 54)
    return result
  }

  public static var dimension: Int { return 5 }

  public static var standardBasis: [Self] {
    var result = Array(repeating: Self.zero, count: 5)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[0].s0 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[1].s1 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[2].s2 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[3].s3 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[4].s4 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 64)
    return result
  }

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    var index = scalars.startIndex
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s0 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s1 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s2 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s3 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s4 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 73)
  }

  public var scalars: [Double] {
    return [s0, s1, s2, s3, s4]
  }
}

extension Vector5: ElementaryFunctions {}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 10)

/// An element of R^6, with Euclidean inner product.
public struct Vector6: KeyPathIterable {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s0: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s1: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s2: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s3: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s4: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s5: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 16)

  @differentiable
  public init(_ s0: Double, _ s1: Double, _ s2: Double, _ s3: Double, _ s4: Double, _ s5: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s0 = s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s1 = s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s2 = s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s3 = s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s4 = s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s5 = s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 22)
  }
}

/// Conformance to EuclideanVectorN
extension Vector6: AdditiveArithmetic, EuclideanVectorN {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s0 += rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s1 += rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s2 += rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s3 += rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s4 += rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s5 += rhs.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 32)
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s0 -= rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s1 -= rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s2 -= rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s3 -= rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s4 -= rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s5 -= rhs.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 39)
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s0 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s1 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s2 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s3 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s4 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s5 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 46)
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s0 * other.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s1 * other.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s2 * other.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s3 * other.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s4 * other.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s5 * other.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 54)
    return result
  }

  public static var dimension: Int { return 6 }

  public static var standardBasis: [Self] {
    var result = Array(repeating: Self.zero, count: 6)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[0].s0 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[1].s1 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[2].s2 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[3].s3 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[4].s4 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[5].s5 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 64)
    return result
  }

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    var index = scalars.startIndex
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s0 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s1 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s2 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s3 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s4 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s5 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 73)
  }

  public var scalars: [Double] {
    return [s0, s1, s2, s3, s4, s5]
  }
}

extension Vector6: ElementaryFunctions {}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 10)

/// An element of R^7, with Euclidean inner product.
public struct Vector7: KeyPathIterable {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s0: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s1: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s2: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s3: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s4: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s5: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s6: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 16)

  @differentiable
  public init(_ s0: Double, _ s1: Double, _ s2: Double, _ s3: Double, _ s4: Double, _ s5: Double, _ s6: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s0 = s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s1 = s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s2 = s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s3 = s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s4 = s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s5 = s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s6 = s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 22)
  }
}

/// Conformance to EuclideanVectorN
extension Vector7: AdditiveArithmetic, EuclideanVectorN {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s0 += rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s1 += rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s2 += rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s3 += rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s4 += rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s5 += rhs.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s6 += rhs.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 32)
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s0 -= rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s1 -= rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s2 -= rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s3 -= rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s4 -= rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s5 -= rhs.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s6 -= rhs.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 39)
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s0 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s1 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s2 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s3 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s4 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s5 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s6 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 46)
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s0 * other.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s1 * other.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s2 * other.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s3 * other.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s4 * other.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s5 * other.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s6 * other.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 54)
    return result
  }

  public static var dimension: Int { return 7 }

  public static var standardBasis: [Self] {
    var result = Array(repeating: Self.zero, count: 7)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[0].s0 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[1].s1 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[2].s2 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[3].s3 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[4].s4 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[5].s5 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[6].s6 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 64)
    return result
  }

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    var index = scalars.startIndex
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s0 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s1 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s2 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s3 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s4 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s5 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s6 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 73)
  }

  public var scalars: [Double] {
    return [s0, s1, s2, s3, s4, s5, s6]
  }
}

extension Vector7: ElementaryFunctions {}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 10)

/// An element of R^8, with Euclidean inner product.
public struct Vector8: KeyPathIterable {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s0: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s1: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s2: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s3: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s4: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s5: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s6: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s7: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 16)

  @differentiable
  public init(_ s0: Double, _ s1: Double, _ s2: Double, _ s3: Double, _ s4: Double, _ s5: Double, _ s6: Double, _ s7: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s0 = s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s1 = s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s2 = s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s3 = s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s4 = s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s5 = s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s6 = s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s7 = s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 22)
  }
}

/// Conformance to EuclideanVectorN
extension Vector8: AdditiveArithmetic, EuclideanVectorN {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s0 += rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s1 += rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s2 += rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s3 += rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s4 += rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s5 += rhs.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s6 += rhs.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s7 += rhs.s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 32)
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s0 -= rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s1 -= rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s2 -= rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s3 -= rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s4 -= rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s5 -= rhs.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s6 -= rhs.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s7 -= rhs.s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 39)
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s0 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s1 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s2 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s3 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s4 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s5 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s6 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s7 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 46)
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s0 * other.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s1 * other.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s2 * other.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s3 * other.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s4 * other.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s5 * other.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s6 * other.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s7 * other.s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 54)
    return result
  }

  public static var dimension: Int { return 8 }

  public static var standardBasis: [Self] {
    var result = Array(repeating: Self.zero, count: 8)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[0].s0 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[1].s1 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[2].s2 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[3].s3 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[4].s4 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[5].s5 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[6].s6 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[7].s7 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 64)
    return result
  }

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    var index = scalars.startIndex
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s0 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s1 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s2 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s3 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s4 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s5 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s6 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s7 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 73)
  }

  public var scalars: [Double] {
    return [s0, s1, s2, s3, s4, s5, s6, s7]
  }
}

extension Vector8: ElementaryFunctions {}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 10)

/// An element of R^9, with Euclidean inner product.
public struct Vector9: KeyPathIterable {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s0: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s1: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s2: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s3: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s4: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s5: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s6: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s7: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s8: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 16)

  @differentiable
  public init(_ s0: Double, _ s1: Double, _ s2: Double, _ s3: Double, _ s4: Double, _ s5: Double, _ s6: Double, _ s7: Double, _ s8: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s0 = s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s1 = s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s2 = s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s3 = s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s4 = s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s5 = s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s6 = s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s7 = s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s8 = s8
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 22)
  }
}

/// Conformance to EuclideanVectorN
extension Vector9: AdditiveArithmetic, EuclideanVectorN {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s0 += rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s1 += rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s2 += rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s3 += rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s4 += rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s5 += rhs.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s6 += rhs.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s7 += rhs.s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s8 += rhs.s8
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 32)
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s0 -= rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s1 -= rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s2 -= rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s3 -= rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s4 -= rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s5 -= rhs.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s6 -= rhs.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s7 -= rhs.s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s8 -= rhs.s8
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 39)
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s0 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s1 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s2 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s3 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s4 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s5 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s6 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s7 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s8 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 46)
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s0 * other.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s1 * other.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s2 * other.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s3 * other.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s4 * other.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s5 * other.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s6 * other.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s7 * other.s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s8 * other.s8
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 54)
    return result
  }

  public static var dimension: Int { return 9 }

  public static var standardBasis: [Self] {
    var result = Array(repeating: Self.zero, count: 9)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[0].s0 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[1].s1 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[2].s2 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[3].s3 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[4].s4 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[5].s5 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[6].s6 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[7].s7 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[8].s8 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 64)
    return result
  }

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    var index = scalars.startIndex
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s0 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s1 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s2 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s3 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s4 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s5 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s6 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s7 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s8 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 73)
  }

  public var scalars: [Double] {
    return [s0, s1, s2, s3, s4, s5, s6, s7, s8]
  }
}

extension Vector9: ElementaryFunctions {}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 10)

/// An element of R^10, with Euclidean inner product.
public struct Vector10: KeyPathIterable {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s0: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s1: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s2: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s3: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s4: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s5: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s6: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s7: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s8: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s9: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 16)

  @differentiable
  public init(_ s0: Double, _ s1: Double, _ s2: Double, _ s3: Double, _ s4: Double, _ s5: Double, _ s6: Double, _ s7: Double, _ s8: Double, _ s9: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s0 = s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s1 = s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s2 = s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s3 = s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s4 = s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s5 = s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s6 = s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s7 = s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s8 = s8
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s9 = s9
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 22)
  }
}

/// Conformance to EuclideanVectorN
extension Vector10: AdditiveArithmetic, EuclideanVectorN {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s0 += rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s1 += rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s2 += rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s3 += rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s4 += rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s5 += rhs.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s6 += rhs.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s7 += rhs.s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s8 += rhs.s8
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s9 += rhs.s9
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 32)
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s0 -= rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s1 -= rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s2 -= rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s3 -= rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s4 -= rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s5 -= rhs.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s6 -= rhs.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s7 -= rhs.s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s8 -= rhs.s8
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s9 -= rhs.s9
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 39)
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s0 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s1 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s2 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s3 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s4 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s5 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s6 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s7 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s8 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s9 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 46)
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s0 * other.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s1 * other.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s2 * other.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s3 * other.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s4 * other.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s5 * other.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s6 * other.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s7 * other.s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s8 * other.s8
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s9 * other.s9
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 54)
    return result
  }

  public static var dimension: Int { return 10 }

  public static var standardBasis: [Self] {
    var result = Array(repeating: Self.zero, count: 10)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[0].s0 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[1].s1 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[2].s2 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[3].s3 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[4].s4 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[5].s5 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[6].s6 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[7].s7 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[8].s8 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[9].s9 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 64)
    return result
  }

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    var index = scalars.startIndex
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s0 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s1 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s2 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s3 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s4 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s5 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s6 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s7 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s8 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s9 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 73)
  }

  public var scalars: [Double] {
    return [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9]
  }
}

extension Vector10: ElementaryFunctions {}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 10)

/// An element of R^11, with Euclidean inner product.
public struct Vector11: KeyPathIterable {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s0: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s1: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s2: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s3: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s4: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s5: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s6: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s7: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s8: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s9: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s10: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 16)

  @differentiable
  public init(_ s0: Double, _ s1: Double, _ s2: Double, _ s3: Double, _ s4: Double, _ s5: Double, _ s6: Double, _ s7: Double, _ s8: Double, _ s9: Double, _ s10: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s0 = s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s1 = s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s2 = s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s3 = s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s4 = s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s5 = s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s6 = s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s7 = s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s8 = s8
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s9 = s9
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s10 = s10
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 22)
  }
}

/// Conformance to EuclideanVectorN
extension Vector11: AdditiveArithmetic, EuclideanVectorN {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s0 += rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s1 += rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s2 += rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s3 += rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s4 += rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s5 += rhs.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s6 += rhs.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s7 += rhs.s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s8 += rhs.s8
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s9 += rhs.s9
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s10 += rhs.s10
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 32)
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s0 -= rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s1 -= rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s2 -= rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s3 -= rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s4 -= rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s5 -= rhs.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s6 -= rhs.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s7 -= rhs.s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s8 -= rhs.s8
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s9 -= rhs.s9
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s10 -= rhs.s10
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 39)
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s0 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s1 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s2 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s3 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s4 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s5 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s6 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s7 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s8 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s9 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s10 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 46)
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s0 * other.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s1 * other.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s2 * other.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s3 * other.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s4 * other.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s5 * other.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s6 * other.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s7 * other.s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s8 * other.s8
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s9 * other.s9
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s10 * other.s10
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 54)
    return result
  }

  public static var dimension: Int { return 11 }

  public static var standardBasis: [Self] {
    var result = Array(repeating: Self.zero, count: 11)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[0].s0 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[1].s1 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[2].s2 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[3].s3 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[4].s4 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[5].s5 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[6].s6 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[7].s7 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[8].s8 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[9].s9 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[10].s10 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 64)
    return result
  }

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    var index = scalars.startIndex
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s0 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s1 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s2 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s3 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s4 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s5 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s6 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s7 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s8 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s9 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s10 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 73)
  }

  public var scalars: [Double] {
    return [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]
  }
}

extension Vector11: ElementaryFunctions {}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 10)

/// An element of R^12, with Euclidean inner product.
public struct Vector12: KeyPathIterable {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s0: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s1: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s2: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s3: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s4: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s5: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s6: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s7: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s8: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s9: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s10: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 14)
  @differentiable public var s11: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 16)

  @differentiable
  public init(_ s0: Double, _ s1: Double, _ s2: Double, _ s3: Double, _ s4: Double, _ s5: Double, _ s6: Double, _ s7: Double, _ s8: Double, _ s9: Double, _ s10: Double, _ s11: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s0 = s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s1 = s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s2 = s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s3 = s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s4 = s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s5 = s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s6 = s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s7 = s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s8 = s8
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s9 = s9
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s10 = s10
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 20)
    self.s11 = s11
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 22)
  }
}

/// Conformance to EuclideanVectorN
extension Vector12: AdditiveArithmetic, EuclideanVectorN {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s0 += rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s1 += rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s2 += rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s3 += rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s4 += rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s5 += rhs.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s6 += rhs.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s7 += rhs.s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s8 += rhs.s8
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s9 += rhs.s9
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s10 += rhs.s10
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 30)
    lhs.s11 += rhs.s11
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 32)
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s0 -= rhs.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s1 -= rhs.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s2 -= rhs.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s3 -= rhs.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s4 -= rhs.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s5 -= rhs.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s6 -= rhs.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s7 -= rhs.s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s8 -= rhs.s8
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s9 -= rhs.s9
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s10 -= rhs.s10
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 37)
    lhs.s11 -= rhs.s11
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 39)
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s0 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s1 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s2 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s3 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s4 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s5 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s6 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s7 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s8 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s9 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s10 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 44)
    lhs.s11 *= rhs
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 46)
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s0 * other.s0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s1 * other.s1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s2 * other.s2
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s3 * other.s3
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s4 * other.s4
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s5 * other.s5
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s6 * other.s6
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s7 * other.s7
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s8 * other.s8
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s9 * other.s9
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s10 * other.s10
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 52)
    result += self.s11 * other.s11
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 54)
    return result
  }

  public static var dimension: Int { return 12 }

  public static var standardBasis: [Self] {
    var result = Array(repeating: Self.zero, count: 12)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[0].s0 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[1].s1 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[2].s2 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[3].s3 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[4].s4 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[5].s5 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[6].s6 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[7].s7 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[8].s8 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[9].s9 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[10].s10 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 62)
    result[11].s11 = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 64)
    return result
  }

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    var index = scalars.startIndex
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s0 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s1 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s2 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s3 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s4 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s5 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s6 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s7 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s8 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s9 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s10 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 70)
    self.s11 = scalars[index]
    index = scalars.index(after: index)
// ###sourceLocation(file: "Sources/SwiftFusion/Core/VectorN.swift.gyb", line: 73)
  }

  public var scalars: [Double] {
    return [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11]
  }
}

extension Vector12: ElementaryFunctions {}

