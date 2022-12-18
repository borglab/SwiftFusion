// WARNING: This is a generated file. Do not edit it. Instead, edit the corresponding ".gyb" file.
// See "generate.sh" in the root of this repository for instructions how to regenerate files.

// import TensorFlow
import _Differentiation


/// An element of R^1, with Euclidean inner product.
public struct Vector1: Codable, KeyPathIterable {
  @differentiable(reverse) public var x: Double

  @differentiable(reverse)
  public init(_ x: Double) {
    self.x = x
  }
}

/// Conformance to Vector
extension Vector1: AdditiveArithmetic, Vector {
  @differentiable(reverse)
  public static func += (_ lhs: inout Self, _ rhs: Self) {
    lhs.x += rhs.x
  }

  @differentiable(reverse)
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
    lhs.x -= rhs.x
  }

  @differentiable(reverse)
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
    lhs.x *= rhs
  }

  @differentiable(reverse)
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
    result += self.x * other.x
    return result
  }

  public var dimension: Int { return 1 }

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    var index = scalars.startIndex
    self.x = scalars[index]
    index = scalars.index(after: index)
  }

  /// A type that can represent all of this vector's scalar values in a standard basis.
  public struct Scalars: RandomAccessCollection, MutableCollection {
    // Deduction of Indices fails without an explicit declaration.
    /// A type that can represent all the indices of elements in this collection.
    public typealias Indices = Range<Int>

    /// The vector whose scalars are reflected by `self`.
    internal var base: Vector1

    /// The position of the first element, or `endIndex` if `self.isEmpty`.
    public var startIndex: Int { 0 }
    
    /// The position one step beyond the last contained element.
    public var endIndex: Int { base.dimension }

    /// Accesses the scalar at `i`.
    public subscript(i: Int) -> Double {
      get {
        precondition(i >= 0 && i < endIndex)
        return withUnsafePointer(to: self) {
          UnsafeRawPointer($0).assumingMemoryBound(to: Double.self)[i]
        }
      }
      _modify {
        precondition(i >= 0 && i < endIndex)
        let p = withUnsafeMutablePointer(to: &self) { $0 }
        let q = UnsafeMutableRawPointer(p).assumingMemoryBound(to: Double.self)
        defer { _fixLifetime(self) }
        yield &q[i]
      }
    }
  }
  
  /// This vector's scalar values in a standard basis.
  public var scalars: Scalars {
    get { .init(base: self) }
    set { self = newValue.base  }
  }
}

extension Vector1: FixedSizeVector {
  public static var dimension: Int { return 1 }
}


/// An element of R^2, with Euclidean inner product.
public struct Vector2: Codable, KeyPathIterable {
  @differentiable(reverse) public var x: Double
  @differentiable(reverse) public var y: Double

  @differentiable(reverse)
  public init(_ x: Double, _ y: Double) {
    self.x = x
    self.y = y
  }
}

/// Conformance to Vector
extension Vector2: AdditiveArithmetic, Vector {
  @differentiable(reverse)
  public static func += (_ lhs: inout Self, _ rhs: Self) {
    lhs.x += rhs.x
    lhs.y += rhs.y
  }

  @differentiable(reverse)
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
    lhs.x -= rhs.x
    lhs.y -= rhs.y
  }

  @differentiable(reverse)
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
    lhs.x *= rhs
    lhs.y *= rhs
  }

  @differentiable(reverse)
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
    result += self.x * other.x
    result += self.y * other.y
    return result
  }

  public var dimension: Int { return 2 }

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    var index = scalars.startIndex
    self.x = scalars[index]
    index = scalars.index(after: index)
    self.y = scalars[index]
    index = scalars.index(after: index)
  }

  /// A type that can represent all of this vector's scalar values in a standard basis.
  public struct Scalars: RandomAccessCollection, MutableCollection {
    // Deduction of Indices fails without an explicit declaration.
    /// A type that can represent all the indices of elements in this collection.
    public typealias Indices = Range<Int>

    /// The vector whose scalars are reflected by `self`.
    internal var base: Vector2

    /// The position of the first element, or `endIndex` if `self.isEmpty`.
    public var startIndex: Int { 0 }
    
    /// The position one step beyond the last contained element.
    public var endIndex: Int { base.dimension }

    /// Accesses the scalar at `i`.
    public subscript(i: Int) -> Double {
      get {
        precondition(i >= 0 && i < endIndex)
        return withUnsafePointer(to: self) {
          UnsafeRawPointer($0).assumingMemoryBound(to: Double.self)[i]
        }
      }
      _modify {
        precondition(i >= 0 && i < endIndex)
        let p = withUnsafeMutablePointer(to: &self) { $0 }
        let q = UnsafeMutableRawPointer(p).assumingMemoryBound(to: Double.self)
        defer { _fixLifetime(self) }
        yield &q[i]
      }
    }
  }
  
  /// This vector's scalar values in a standard basis.
  public var scalars: Scalars {
    get { .init(base: self) }
    set { self = newValue.base  }
  }
}

extension Vector2: FixedSizeVector {
  public static var dimension: Int { return 2 }
}


/// An element of R^3, with Euclidean inner product.
public struct Vector3: Codable, KeyPathIterable {
  @differentiable(reverse) public var x: Double
  @differentiable(reverse) public var y: Double
  @differentiable(reverse) public var z: Double

  @differentiable(reverse)
  public init(_ x: Double, _ y: Double, _ z: Double) {
    self.x = x
    self.y = y
    self.z = z
  }
}

/// Conformance to Vector
extension Vector3: AdditiveArithmetic, Vector {
  @differentiable(reverse)
  public static func += (_ lhs: inout Self, _ rhs: Self) {
    lhs.x += rhs.x
    lhs.y += rhs.y
    lhs.z += rhs.z
  }

  @differentiable(reverse)
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
    lhs.x -= rhs.x
    lhs.y -= rhs.y
    lhs.z -= rhs.z
  }

  @differentiable(reverse)
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
    lhs.x *= rhs
    lhs.y *= rhs
    lhs.z *= rhs
  }

  @differentiable(reverse)
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
    result += self.x * other.x
    result += self.y * other.y
    result += self.z * other.z
    return result
  }

  public var dimension: Int { return 3 }

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    var index = scalars.startIndex
    self.x = scalars[index]
    index = scalars.index(after: index)
    self.y = scalars[index]
    index = scalars.index(after: index)
    self.z = scalars[index]
    index = scalars.index(after: index)
  }

  /// A type that can represent all of this vector's scalar values in a standard basis.
  public struct Scalars: RandomAccessCollection, MutableCollection {
    // Deduction of Indices fails without an explicit declaration.
    /// A type that can represent all the indices of elements in this collection.
    public typealias Indices = Range<Int>

    /// The vector whose scalars are reflected by `self`.
    internal var base: Vector3

    /// The position of the first element, or `endIndex` if `self.isEmpty`.
    public var startIndex: Int { 0 }
    
    /// The position one step beyond the last contained element.
    public var endIndex: Int { base.dimension }

    /// Accesses the scalar at `i`.
    public subscript(i: Int) -> Double {
      get {
        precondition(i >= 0 && i < endIndex)
        return withUnsafePointer(to: self) {
          UnsafeRawPointer($0).assumingMemoryBound(to: Double.self)[i]
        }
      }
      _modify {
        precondition(i >= 0 && i < endIndex)
        let p = withUnsafeMutablePointer(to: &self) { $0 }
        let q = UnsafeMutableRawPointer(p).assumingMemoryBound(to: Double.self)
        defer { _fixLifetime(self) }
        yield &q[i]
      }
    }
  }
  
  /// This vector's scalar values in a standard basis.
  public var scalars: Scalars {
    get { .init(base: self) }
    set { self = newValue.base  }
  }
}

extension Vector3: FixedSizeVector {
  public static var dimension: Int { return 3 }
}


/// An element of R^4, with Euclidean inner product.
public struct Vector4: Codable, KeyPathIterable {
  @differentiable(reverse) public var s0: Double
  @differentiable(reverse) public var s1: Double
  @differentiable(reverse) public var s2: Double
  @differentiable(reverse) public var s3: Double

  @differentiable(reverse)
  public init(_ s0: Double, _ s1: Double, _ s2: Double, _ s3: Double) {
    self.s0 = s0
    self.s1 = s1
    self.s2 = s2
    self.s3 = s3
  }
}

/// Conformance to Vector
extension Vector4: AdditiveArithmetic, Vector {
  @differentiable(reverse)
  public static func += (_ lhs: inout Self, _ rhs: Self) {
    lhs.s0 += rhs.s0
    lhs.s1 += rhs.s1
    lhs.s2 += rhs.s2
    lhs.s3 += rhs.s3
  }

  @differentiable(reverse)
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
    lhs.s0 -= rhs.s0
    lhs.s1 -= rhs.s1
    lhs.s2 -= rhs.s2
    lhs.s3 -= rhs.s3
  }

  @differentiable(reverse)
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
    lhs.s0 *= rhs
    lhs.s1 *= rhs
    lhs.s2 *= rhs
    lhs.s3 *= rhs
  }

  @differentiable(reverse)
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
    result += self.s0 * other.s0
    result += self.s1 * other.s1
    result += self.s2 * other.s2
    result += self.s3 * other.s3
    return result
  }

  public var dimension: Int { return 4 }

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    var index = scalars.startIndex
    self.s0 = scalars[index]
    index = scalars.index(after: index)
    self.s1 = scalars[index]
    index = scalars.index(after: index)
    self.s2 = scalars[index]
    index = scalars.index(after: index)
    self.s3 = scalars[index]
    index = scalars.index(after: index)
  }

  /// A type that can represent all of this vector's scalar values in a standard basis.
  public struct Scalars: RandomAccessCollection, MutableCollection {
    // Deduction of Indices fails without an explicit declaration.
    /// A type that can represent all the indices of elements in this collection.
    public typealias Indices = Range<Int>

    /// The vector whose scalars are reflected by `self`.
    internal var base: Vector4

    /// The position of the first element, or `endIndex` if `self.isEmpty`.
    public var startIndex: Int { 0 }
    
    /// The position one step beyond the last contained element.
    public var endIndex: Int { base.dimension }

    /// Accesses the scalar at `i`.
    public subscript(i: Int) -> Double {
      get {
        precondition(i >= 0 && i < endIndex)
        return withUnsafePointer(to: self) {
          UnsafeRawPointer($0).assumingMemoryBound(to: Double.self)[i]
        }
      }
      _modify {
        precondition(i >= 0 && i < endIndex)
        let p = withUnsafeMutablePointer(to: &self) { $0 }
        let q = UnsafeMutableRawPointer(p).assumingMemoryBound(to: Double.self)
        defer { _fixLifetime(self) }
        yield &q[i]
      }
    }
  }
  
  /// This vector's scalar values in a standard basis.
  public var scalars: Scalars {
    get { .init(base: self) }
    set { self = newValue.base  }
  }
}

extension Vector4: FixedSizeVector {
  public static var dimension: Int { return 4 }
}


/// An element of R^5, with Euclidean inner product.
public struct Vector5: Codable, KeyPathIterable {
  @differentiable(reverse) public var s0: Double
  @differentiable(reverse) public var s1: Double
  @differentiable(reverse) public var s2: Double
  @differentiable(reverse) public var s3: Double
  @differentiable(reverse) public var s4: Double

  @differentiable(reverse)
  public init(_ s0: Double, _ s1: Double, _ s2: Double, _ s3: Double, _ s4: Double) {
    self.s0 = s0
    self.s1 = s1
    self.s2 = s2
    self.s3 = s3
    self.s4 = s4
  }
}

/// Conformance to Vector
extension Vector5: AdditiveArithmetic, Vector {
  @differentiable(reverse)
  public static func += (_ lhs: inout Self, _ rhs: Self) {
    lhs.s0 += rhs.s0
    lhs.s1 += rhs.s1
    lhs.s2 += rhs.s2
    lhs.s3 += rhs.s3
    lhs.s4 += rhs.s4
  }

  @differentiable(reverse)
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
    lhs.s0 -= rhs.s0
    lhs.s1 -= rhs.s1
    lhs.s2 -= rhs.s2
    lhs.s3 -= rhs.s3
    lhs.s4 -= rhs.s4
  }

  @differentiable(reverse)
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
    lhs.s0 *= rhs
    lhs.s1 *= rhs
    lhs.s2 *= rhs
    lhs.s3 *= rhs
    lhs.s4 *= rhs
  }

  @differentiable(reverse)
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
    result += self.s0 * other.s0
    result += self.s1 * other.s1
    result += self.s2 * other.s2
    result += self.s3 * other.s3
    result += self.s4 * other.s4
    return result
  }

  public var dimension: Int { return 5 }

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    var index = scalars.startIndex
    self.s0 = scalars[index]
    index = scalars.index(after: index)
    self.s1 = scalars[index]
    index = scalars.index(after: index)
    self.s2 = scalars[index]
    index = scalars.index(after: index)
    self.s3 = scalars[index]
    index = scalars.index(after: index)
    self.s4 = scalars[index]
    index = scalars.index(after: index)
  }

  /// A type that can represent all of this vector's scalar values in a standard basis.
  public struct Scalars: RandomAccessCollection, MutableCollection {
    // Deduction of Indices fails without an explicit declaration.
    /// A type that can represent all the indices of elements in this collection.
    public typealias Indices = Range<Int>

    /// The vector whose scalars are reflected by `self`.
    internal var base: Vector5

    /// The position of the first element, or `endIndex` if `self.isEmpty`.
    public var startIndex: Int { 0 }
    
    /// The position one step beyond the last contained element.
    public var endIndex: Int { base.dimension }

    /// Accesses the scalar at `i`.
    public subscript(i: Int) -> Double {
      get {
        precondition(i >= 0 && i < endIndex)
        return withUnsafePointer(to: self) {
          UnsafeRawPointer($0).assumingMemoryBound(to: Double.self)[i]
        }
      }
      _modify {
        precondition(i >= 0 && i < endIndex)
        let p = withUnsafeMutablePointer(to: &self) { $0 }
        let q = UnsafeMutableRawPointer(p).assumingMemoryBound(to: Double.self)
        defer { _fixLifetime(self) }
        yield &q[i]
      }
    }
  }
  
  /// This vector's scalar values in a standard basis.
  public var scalars: Scalars {
    get { .init(base: self) }
    set { self = newValue.base  }
  }
}

extension Vector5: FixedSizeVector {
  public static var dimension: Int { return 5 }
}


/// An element of R^6, with Euclidean inner product.
public struct Vector6: Codable, KeyPathIterable {
  @differentiable(reverse) public var s0: Double
  @differentiable(reverse) public var s1: Double
  @differentiable(reverse) public var s2: Double
  @differentiable(reverse) public var s3: Double
  @differentiable(reverse) public var s4: Double
  @differentiable(reverse) public var s5: Double

  @differentiable(reverse)
  public init(_ s0: Double, _ s1: Double, _ s2: Double, _ s3: Double, _ s4: Double, _ s5: Double) {
    self.s0 = s0
    self.s1 = s1
    self.s2 = s2
    self.s3 = s3
    self.s4 = s4
    self.s5 = s5
  }
}

/// Conformance to Vector
extension Vector6: AdditiveArithmetic, Vector {
  @differentiable(reverse)
  public static func += (_ lhs: inout Self, _ rhs: Self) {
    lhs.s0 += rhs.s0
    lhs.s1 += rhs.s1
    lhs.s2 += rhs.s2
    lhs.s3 += rhs.s3
    lhs.s4 += rhs.s4
    lhs.s5 += rhs.s5
  }

  @differentiable(reverse)
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
    lhs.s0 -= rhs.s0
    lhs.s1 -= rhs.s1
    lhs.s2 -= rhs.s2
    lhs.s3 -= rhs.s3
    lhs.s4 -= rhs.s4
    lhs.s5 -= rhs.s5
  }

  @differentiable(reverse)
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
    lhs.s0 *= rhs
    lhs.s1 *= rhs
    lhs.s2 *= rhs
    lhs.s3 *= rhs
    lhs.s4 *= rhs
    lhs.s5 *= rhs
  }

  @differentiable(reverse)
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
    result += self.s0 * other.s0
    result += self.s1 * other.s1
    result += self.s2 * other.s2
    result += self.s3 * other.s3
    result += self.s4 * other.s4
    result += self.s5 * other.s5
    return result
  }

  public var dimension: Int { return 6 }

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    var index = scalars.startIndex
    self.s0 = scalars[index]
    index = scalars.index(after: index)
    self.s1 = scalars[index]
    index = scalars.index(after: index)
    self.s2 = scalars[index]
    index = scalars.index(after: index)
    self.s3 = scalars[index]
    index = scalars.index(after: index)
    self.s4 = scalars[index]
    index = scalars.index(after: index)
    self.s5 = scalars[index]
    index = scalars.index(after: index)
  }

  /// A type that can represent all of this vector's scalar values in a standard basis.
  public struct Scalars: RandomAccessCollection, MutableCollection {
    // Deduction of Indices fails without an explicit declaration.
    /// A type that can represent all the indices of elements in this collection.
    public typealias Indices = Range<Int>

    /// The vector whose scalars are reflected by `self`.
    internal var base: Vector6

    /// The position of the first element, or `endIndex` if `self.isEmpty`.
    public var startIndex: Int { 0 }
    
    /// The position one step beyond the last contained element.
    public var endIndex: Int { base.dimension }

    /// Accesses the scalar at `i`.
    public subscript(i: Int) -> Double {
      get {
        precondition(i >= 0 && i < endIndex)
        return withUnsafePointer(to: self) {
          UnsafeRawPointer($0).assumingMemoryBound(to: Double.self)[i]
        }
      }
      _modify {
        precondition(i >= 0 && i < endIndex)
        let p = withUnsafeMutablePointer(to: &self) { $0 }
        let q = UnsafeMutableRawPointer(p).assumingMemoryBound(to: Double.self)
        defer { _fixLifetime(self) }
        yield &q[i]
      }
    }
  }
  
  /// This vector's scalar values in a standard basis.
  public var scalars: Scalars {
    get { .init(base: self) }
    set { self = newValue.base  }
  }
}

extension Vector6: FixedSizeVector {
  public static var dimension: Int { return 6 }
}


/// An element of R^7, with Euclidean inner product.
public struct Vector7: Codable, KeyPathIterable {
  @differentiable(reverse) public var s0: Double
  @differentiable(reverse) public var s1: Double
  @differentiable(reverse) public var s2: Double
  @differentiable(reverse) public var s3: Double
  @differentiable(reverse) public var s4: Double
  @differentiable(reverse) public var s5: Double
  @differentiable(reverse) public var s6: Double

  @differentiable(reverse)
  public init(_ s0: Double, _ s1: Double, _ s2: Double, _ s3: Double, _ s4: Double, _ s5: Double, _ s6: Double) {
    self.s0 = s0
    self.s1 = s1
    self.s2 = s2
    self.s3 = s3
    self.s4 = s4
    self.s5 = s5
    self.s6 = s6
  }
}

/// Conformance to Vector
extension Vector7: AdditiveArithmetic, Vector {
  @differentiable(reverse)
  public static func += (_ lhs: inout Self, _ rhs: Self) {
    lhs.s0 += rhs.s0
    lhs.s1 += rhs.s1
    lhs.s2 += rhs.s2
    lhs.s3 += rhs.s3
    lhs.s4 += rhs.s4
    lhs.s5 += rhs.s5
    lhs.s6 += rhs.s6
  }

  @differentiable(reverse)
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
    lhs.s0 -= rhs.s0
    lhs.s1 -= rhs.s1
    lhs.s2 -= rhs.s2
    lhs.s3 -= rhs.s3
    lhs.s4 -= rhs.s4
    lhs.s5 -= rhs.s5
    lhs.s6 -= rhs.s6
  }

  @differentiable(reverse)
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
    lhs.s0 *= rhs
    lhs.s1 *= rhs
    lhs.s2 *= rhs
    lhs.s3 *= rhs
    lhs.s4 *= rhs
    lhs.s5 *= rhs
    lhs.s6 *= rhs
  }

  @differentiable(reverse)
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
    result += self.s0 * other.s0
    result += self.s1 * other.s1
    result += self.s2 * other.s2
    result += self.s3 * other.s3
    result += self.s4 * other.s4
    result += self.s5 * other.s5
    result += self.s6 * other.s6
    return result
  }

  public var dimension: Int { return 7 }

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    var index = scalars.startIndex
    self.s0 = scalars[index]
    index = scalars.index(after: index)
    self.s1 = scalars[index]
    index = scalars.index(after: index)
    self.s2 = scalars[index]
    index = scalars.index(after: index)
    self.s3 = scalars[index]
    index = scalars.index(after: index)
    self.s4 = scalars[index]
    index = scalars.index(after: index)
    self.s5 = scalars[index]
    index = scalars.index(after: index)
    self.s6 = scalars[index]
    index = scalars.index(after: index)
  }

  /// A type that can represent all of this vector's scalar values in a standard basis.
  public struct Scalars: RandomAccessCollection, MutableCollection {
    // Deduction of Indices fails without an explicit declaration.
    /// A type that can represent all the indices of elements in this collection.
    public typealias Indices = Range<Int>

    /// The vector whose scalars are reflected by `self`.
    internal var base: Vector7

    /// The position of the first element, or `endIndex` if `self.isEmpty`.
    public var startIndex: Int { 0 }
    
    /// The position one step beyond the last contained element.
    public var endIndex: Int { base.dimension }

    /// Accesses the scalar at `i`.
    public subscript(i: Int) -> Double {
      get {
        precondition(i >= 0 && i < endIndex)
        return withUnsafePointer(to: self) {
          UnsafeRawPointer($0).assumingMemoryBound(to: Double.self)[i]
        }
      }
      _modify {
        precondition(i >= 0 && i < endIndex)
        let p = withUnsafeMutablePointer(to: &self) { $0 }
        let q = UnsafeMutableRawPointer(p).assumingMemoryBound(to: Double.self)
        defer { _fixLifetime(self) }
        yield &q[i]
      }
    }
  }
  
  /// This vector's scalar values in a standard basis.
  public var scalars: Scalars {
    get { .init(base: self) }
    set { self = newValue.base  }
  }
}

extension Vector7: FixedSizeVector {
  public static var dimension: Int { return 7 }
}


/// An element of R^8, with Euclidean inner product.
public struct Vector8: Codable, KeyPathIterable {
  @differentiable(reverse) public var s0: Double
  @differentiable(reverse) public var s1: Double
  @differentiable(reverse) public var s2: Double
  @differentiable(reverse) public var s3: Double
  @differentiable(reverse) public var s4: Double
  @differentiable(reverse) public var s5: Double
  @differentiable(reverse) public var s6: Double
  @differentiable(reverse) public var s7: Double

  @differentiable(reverse)
  public init(_ s0: Double, _ s1: Double, _ s2: Double, _ s3: Double, _ s4: Double, _ s5: Double, _ s6: Double, _ s7: Double) {
    self.s0 = s0
    self.s1 = s1
    self.s2 = s2
    self.s3 = s3
    self.s4 = s4
    self.s5 = s5
    self.s6 = s6
    self.s7 = s7
  }
}

/// Conformance to Vector
extension Vector8: AdditiveArithmetic, Vector {
  @differentiable(reverse)
  public static func += (_ lhs: inout Self, _ rhs: Self) {
    lhs.s0 += rhs.s0
    lhs.s1 += rhs.s1
    lhs.s2 += rhs.s2
    lhs.s3 += rhs.s3
    lhs.s4 += rhs.s4
    lhs.s5 += rhs.s5
    lhs.s6 += rhs.s6
    lhs.s7 += rhs.s7
  }

  @differentiable(reverse)
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
    lhs.s0 -= rhs.s0
    lhs.s1 -= rhs.s1
    lhs.s2 -= rhs.s2
    lhs.s3 -= rhs.s3
    lhs.s4 -= rhs.s4
    lhs.s5 -= rhs.s5
    lhs.s6 -= rhs.s6
    lhs.s7 -= rhs.s7
  }

  @differentiable(reverse)
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
    lhs.s0 *= rhs
    lhs.s1 *= rhs
    lhs.s2 *= rhs
    lhs.s3 *= rhs
    lhs.s4 *= rhs
    lhs.s5 *= rhs
    lhs.s6 *= rhs
    lhs.s7 *= rhs
  }

  @differentiable(reverse)
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
    result += self.s0 * other.s0
    result += self.s1 * other.s1
    result += self.s2 * other.s2
    result += self.s3 * other.s3
    result += self.s4 * other.s4
    result += self.s5 * other.s5
    result += self.s6 * other.s6
    result += self.s7 * other.s7
    return result
  }

  public var dimension: Int { return 8 }

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    var index = scalars.startIndex
    self.s0 = scalars[index]
    index = scalars.index(after: index)
    self.s1 = scalars[index]
    index = scalars.index(after: index)
    self.s2 = scalars[index]
    index = scalars.index(after: index)
    self.s3 = scalars[index]
    index = scalars.index(after: index)
    self.s4 = scalars[index]
    index = scalars.index(after: index)
    self.s5 = scalars[index]
    index = scalars.index(after: index)
    self.s6 = scalars[index]
    index = scalars.index(after: index)
    self.s7 = scalars[index]
    index = scalars.index(after: index)
  }

  /// A type that can represent all of this vector's scalar values in a standard basis.
  public struct Scalars: RandomAccessCollection, MutableCollection {
    // Deduction of Indices fails without an explicit declaration.
    /// A type that can represent all the indices of elements in this collection.
    public typealias Indices = Range<Int>

    /// The vector whose scalars are reflected by `self`.
    internal var base: Vector8

    /// The position of the first element, or `endIndex` if `self.isEmpty`.
    public var startIndex: Int { 0 }
    
    /// The position one step beyond the last contained element.
    public var endIndex: Int { base.dimension }

    /// Accesses the scalar at `i`.
    public subscript(i: Int) -> Double {
      get {
        precondition(i >= 0 && i < endIndex)
        return withUnsafePointer(to: self) {
          UnsafeRawPointer($0).assumingMemoryBound(to: Double.self)[i]
        }
      }
      _modify {
        precondition(i >= 0 && i < endIndex)
        let p = withUnsafeMutablePointer(to: &self) { $0 }
        let q = UnsafeMutableRawPointer(p).assumingMemoryBound(to: Double.self)
        defer { _fixLifetime(self) }
        yield &q[i]
      }
    }
  }
  
  /// This vector's scalar values in a standard basis.
  public var scalars: Scalars {
    get { .init(base: self) }
    set { self = newValue.base  }
  }
}

extension Vector8: FixedSizeVector {
  public static var dimension: Int { return 8 }
}


/// An element of R^9, with Euclidean inner product.
public struct Vector9: Codable, KeyPathIterable {
  @differentiable(reverse) public var s0: Double
  @differentiable(reverse) public var s1: Double
  @differentiable(reverse) public var s2: Double
  @differentiable(reverse) public var s3: Double
  @differentiable(reverse) public var s4: Double
  @differentiable(reverse) public var s5: Double
  @differentiable(reverse) public var s6: Double
  @differentiable(reverse) public var s7: Double
  @differentiable(reverse) public var s8: Double

  @differentiable(reverse)
  public init(_ s0: Double, _ s1: Double, _ s2: Double, _ s3: Double, _ s4: Double, _ s5: Double, _ s6: Double, _ s7: Double, _ s8: Double) {
    self.s0 = s0
    self.s1 = s1
    self.s2 = s2
    self.s3 = s3
    self.s4 = s4
    self.s5 = s5
    self.s6 = s6
    self.s7 = s7
    self.s8 = s8
  }
}

/// Conformance to Vector
extension Vector9: AdditiveArithmetic, Vector {
  @differentiable(reverse)
  public static func += (_ lhs: inout Self, _ rhs: Self) {
    lhs.s0 += rhs.s0
    lhs.s1 += rhs.s1
    lhs.s2 += rhs.s2
    lhs.s3 += rhs.s3
    lhs.s4 += rhs.s4
    lhs.s5 += rhs.s5
    lhs.s6 += rhs.s6
    lhs.s7 += rhs.s7
    lhs.s8 += rhs.s8
  }

  @differentiable(reverse)
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
    lhs.s0 -= rhs.s0
    lhs.s1 -= rhs.s1
    lhs.s2 -= rhs.s2
    lhs.s3 -= rhs.s3
    lhs.s4 -= rhs.s4
    lhs.s5 -= rhs.s5
    lhs.s6 -= rhs.s6
    lhs.s7 -= rhs.s7
    lhs.s8 -= rhs.s8
  }

  @differentiable(reverse)
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
    lhs.s0 *= rhs
    lhs.s1 *= rhs
    lhs.s2 *= rhs
    lhs.s3 *= rhs
    lhs.s4 *= rhs
    lhs.s5 *= rhs
    lhs.s6 *= rhs
    lhs.s7 *= rhs
    lhs.s8 *= rhs
  }

  @differentiable(reverse)
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
    result += self.s0 * other.s0
    result += self.s1 * other.s1
    result += self.s2 * other.s2
    result += self.s3 * other.s3
    result += self.s4 * other.s4
    result += self.s5 * other.s5
    result += self.s6 * other.s6
    result += self.s7 * other.s7
    result += self.s8 * other.s8
    return result
  }

  public var dimension: Int { return 9 }

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    var index = scalars.startIndex
    self.s0 = scalars[index]
    index = scalars.index(after: index)
    self.s1 = scalars[index]
    index = scalars.index(after: index)
    self.s2 = scalars[index]
    index = scalars.index(after: index)
    self.s3 = scalars[index]
    index = scalars.index(after: index)
    self.s4 = scalars[index]
    index = scalars.index(after: index)
    self.s5 = scalars[index]
    index = scalars.index(after: index)
    self.s6 = scalars[index]
    index = scalars.index(after: index)
    self.s7 = scalars[index]
    index = scalars.index(after: index)
    self.s8 = scalars[index]
    index = scalars.index(after: index)
  }

  /// A type that can represent all of this vector's scalar values in a standard basis.
  public struct Scalars: RandomAccessCollection, MutableCollection {
    // Deduction of Indices fails without an explicit declaration.
    /// A type that can represent all the indices of elements in this collection.
    public typealias Indices = Range<Int>

    /// The vector whose scalars are reflected by `self`.
    internal var base: Vector9

    /// The position of the first element, or `endIndex` if `self.isEmpty`.
    public var startIndex: Int { 0 }
    
    /// The position one step beyond the last contained element.
    public var endIndex: Int { base.dimension }

    /// Accesses the scalar at `i`.
    public subscript(i: Int) -> Double {
      get {
        precondition(i >= 0 && i < endIndex)
        return withUnsafePointer(to: self) {
          UnsafeRawPointer($0).assumingMemoryBound(to: Double.self)[i]
        }
      }
      _modify {
        precondition(i >= 0 && i < endIndex)
        let p = withUnsafeMutablePointer(to: &self) { $0 }
        let q = UnsafeMutableRawPointer(p).assumingMemoryBound(to: Double.self)
        defer { _fixLifetime(self) }
        yield &q[i]
      }
    }
  }
  
  /// This vector's scalar values in a standard basis.
  public var scalars: Scalars {
    get { .init(base: self) }
    set { self = newValue.base  }
  }
}

extension Vector9: FixedSizeVector {
  public static var dimension: Int { return 9 }
}


/// An element of R^10, with Euclidean inner product.
public struct Vector10: Codable, KeyPathIterable {
  @differentiable(reverse) public var s0: Double
  @differentiable(reverse) public var s1: Double
  @differentiable(reverse) public var s2: Double
  @differentiable(reverse) public var s3: Double
  @differentiable(reverse) public var s4: Double
  @differentiable(reverse) public var s5: Double
  @differentiable(reverse) public var s6: Double
  @differentiable(reverse) public var s7: Double
  @differentiable(reverse) public var s8: Double
  @differentiable(reverse) public var s9: Double

  @differentiable(reverse)
  public init(_ s0: Double, _ s1: Double, _ s2: Double, _ s3: Double, _ s4: Double, _ s5: Double, _ s6: Double, _ s7: Double, _ s8: Double, _ s9: Double) {
    self.s0 = s0
    self.s1 = s1
    self.s2 = s2
    self.s3 = s3
    self.s4 = s4
    self.s5 = s5
    self.s6 = s6
    self.s7 = s7
    self.s8 = s8
    self.s9 = s9
  }
}

/// Conformance to Vector
extension Vector10: AdditiveArithmetic, Vector {
  @differentiable(reverse)
  public static func += (_ lhs: inout Self, _ rhs: Self) {
    lhs.s0 += rhs.s0
    lhs.s1 += rhs.s1
    lhs.s2 += rhs.s2
    lhs.s3 += rhs.s3
    lhs.s4 += rhs.s4
    lhs.s5 += rhs.s5
    lhs.s6 += rhs.s6
    lhs.s7 += rhs.s7
    lhs.s8 += rhs.s8
    lhs.s9 += rhs.s9
  }

  @differentiable(reverse)
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
    lhs.s0 -= rhs.s0
    lhs.s1 -= rhs.s1
    lhs.s2 -= rhs.s2
    lhs.s3 -= rhs.s3
    lhs.s4 -= rhs.s4
    lhs.s5 -= rhs.s5
    lhs.s6 -= rhs.s6
    lhs.s7 -= rhs.s7
    lhs.s8 -= rhs.s8
    lhs.s9 -= rhs.s9
  }

  @differentiable(reverse)
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
    lhs.s0 *= rhs
    lhs.s1 *= rhs
    lhs.s2 *= rhs
    lhs.s3 *= rhs
    lhs.s4 *= rhs
    lhs.s5 *= rhs
    lhs.s6 *= rhs
    lhs.s7 *= rhs
    lhs.s8 *= rhs
    lhs.s9 *= rhs
  }

  @differentiable(reverse)
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
    result += self.s0 * other.s0
    result += self.s1 * other.s1
    result += self.s2 * other.s2
    result += self.s3 * other.s3
    result += self.s4 * other.s4
    result += self.s5 * other.s5
    result += self.s6 * other.s6
    result += self.s7 * other.s7
    result += self.s8 * other.s8
    result += self.s9 * other.s9
    return result
  }

  public var dimension: Int { return 10 }

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    var index = scalars.startIndex
    self.s0 = scalars[index]
    index = scalars.index(after: index)
    self.s1 = scalars[index]
    index = scalars.index(after: index)
    self.s2 = scalars[index]
    index = scalars.index(after: index)
    self.s3 = scalars[index]
    index = scalars.index(after: index)
    self.s4 = scalars[index]
    index = scalars.index(after: index)
    self.s5 = scalars[index]
    index = scalars.index(after: index)
    self.s6 = scalars[index]
    index = scalars.index(after: index)
    self.s7 = scalars[index]
    index = scalars.index(after: index)
    self.s8 = scalars[index]
    index = scalars.index(after: index)
    self.s9 = scalars[index]
    index = scalars.index(after: index)
  }

  /// A type that can represent all of this vector's scalar values in a standard basis.
  public struct Scalars: RandomAccessCollection, MutableCollection {
    // Deduction of Indices fails without an explicit declaration.
    /// A type that can represent all the indices of elements in this collection.
    public typealias Indices = Range<Int>

    /// The vector whose scalars are reflected by `self`.
    internal var base: Vector10

    /// The position of the first element, or `endIndex` if `self.isEmpty`.
    public var startIndex: Int { 0 }
    
    /// The position one step beyond the last contained element.
    public var endIndex: Int { base.dimension }

    /// Accesses the scalar at `i`.
    public subscript(i: Int) -> Double {
      get {
        precondition(i >= 0 && i < endIndex)
        return withUnsafePointer(to: self) {
          UnsafeRawPointer($0).assumingMemoryBound(to: Double.self)[i]
        }
      }
      _modify {
        precondition(i >= 0 && i < endIndex)
        let p = withUnsafeMutablePointer(to: &self) { $0 }
        let q = UnsafeMutableRawPointer(p).assumingMemoryBound(to: Double.self)
        defer { _fixLifetime(self) }
        yield &q[i]
      }
    }
  }
  
  /// This vector's scalar values in a standard basis.
  public var scalars: Scalars {
    get { .init(base: self) }
    set { self = newValue.base  }
  }
}

extension Vector10: FixedSizeVector {
  public static var dimension: Int { return 10 }
}


/// An element of R^11, with Euclidean inner product.
public struct Vector11: Codable, KeyPathIterable {
  @differentiable(reverse) public var s0: Double
  @differentiable(reverse) public var s1: Double
  @differentiable(reverse) public var s2: Double
  @differentiable(reverse) public var s3: Double
  @differentiable(reverse) public var s4: Double
  @differentiable(reverse) public var s5: Double
  @differentiable(reverse) public var s6: Double
  @differentiable(reverse) public var s7: Double
  @differentiable(reverse) public var s8: Double
  @differentiable(reverse) public var s9: Double
  @differentiable(reverse) public var s10: Double

  @differentiable(reverse)
  public init(_ s0: Double, _ s1: Double, _ s2: Double, _ s3: Double, _ s4: Double, _ s5: Double, _ s6: Double, _ s7: Double, _ s8: Double, _ s9: Double, _ s10: Double) {
    self.s0 = s0
    self.s1 = s1
    self.s2 = s2
    self.s3 = s3
    self.s4 = s4
    self.s5 = s5
    self.s6 = s6
    self.s7 = s7
    self.s8 = s8
    self.s9 = s9
    self.s10 = s10
  }
}

/// Conformance to Vector
extension Vector11: AdditiveArithmetic, Vector {
  @differentiable(reverse)
  public static func += (_ lhs: inout Self, _ rhs: Self) {
    lhs.s0 += rhs.s0
    lhs.s1 += rhs.s1
    lhs.s2 += rhs.s2
    lhs.s3 += rhs.s3
    lhs.s4 += rhs.s4
    lhs.s5 += rhs.s5
    lhs.s6 += rhs.s6
    lhs.s7 += rhs.s7
    lhs.s8 += rhs.s8
    lhs.s9 += rhs.s9
    lhs.s10 += rhs.s10
  }

  @differentiable(reverse)
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
    lhs.s0 -= rhs.s0
    lhs.s1 -= rhs.s1
    lhs.s2 -= rhs.s2
    lhs.s3 -= rhs.s3
    lhs.s4 -= rhs.s4
    lhs.s5 -= rhs.s5
    lhs.s6 -= rhs.s6
    lhs.s7 -= rhs.s7
    lhs.s8 -= rhs.s8
    lhs.s9 -= rhs.s9
    lhs.s10 -= rhs.s10
  }

  @differentiable(reverse)
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
    lhs.s0 *= rhs
    lhs.s1 *= rhs
    lhs.s2 *= rhs
    lhs.s3 *= rhs
    lhs.s4 *= rhs
    lhs.s5 *= rhs
    lhs.s6 *= rhs
    lhs.s7 *= rhs
    lhs.s8 *= rhs
    lhs.s9 *= rhs
    lhs.s10 *= rhs
  }

  @differentiable(reverse)
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
    result += self.s0 * other.s0
    result += self.s1 * other.s1
    result += self.s2 * other.s2
    result += self.s3 * other.s3
    result += self.s4 * other.s4
    result += self.s5 * other.s5
    result += self.s6 * other.s6
    result += self.s7 * other.s7
    result += self.s8 * other.s8
    result += self.s9 * other.s9
    result += self.s10 * other.s10
    return result
  }

  public var dimension: Int { return 11 }

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    var index = scalars.startIndex
    self.s0 = scalars[index]
    index = scalars.index(after: index)
    self.s1 = scalars[index]
    index = scalars.index(after: index)
    self.s2 = scalars[index]
    index = scalars.index(after: index)
    self.s3 = scalars[index]
    index = scalars.index(after: index)
    self.s4 = scalars[index]
    index = scalars.index(after: index)
    self.s5 = scalars[index]
    index = scalars.index(after: index)
    self.s6 = scalars[index]
    index = scalars.index(after: index)
    self.s7 = scalars[index]
    index = scalars.index(after: index)
    self.s8 = scalars[index]
    index = scalars.index(after: index)
    self.s9 = scalars[index]
    index = scalars.index(after: index)
    self.s10 = scalars[index]
    index = scalars.index(after: index)
  }

  /// A type that can represent all of this vector's scalar values in a standard basis.
  public struct Scalars: RandomAccessCollection, MutableCollection {
    // Deduction of Indices fails without an explicit declaration.
    /// A type that can represent all the indices of elements in this collection.
    public typealias Indices = Range<Int>

    /// The vector whose scalars are reflected by `self`.
    internal var base: Vector11

    /// The position of the first element, or `endIndex` if `self.isEmpty`.
    public var startIndex: Int { 0 }
    
    /// The position one step beyond the last contained element.
    public var endIndex: Int { base.dimension }

    /// Accesses the scalar at `i`.
    public subscript(i: Int) -> Double {
      get {
        precondition(i >= 0 && i < endIndex)
        return withUnsafePointer(to: self) {
          UnsafeRawPointer($0).assumingMemoryBound(to: Double.self)[i]
        }
      }
      _modify {
        precondition(i >= 0 && i < endIndex)
        let p = withUnsafeMutablePointer(to: &self) { $0 }
        let q = UnsafeMutableRawPointer(p).assumingMemoryBound(to: Double.self)
        defer { _fixLifetime(self) }
        yield &q[i]
      }
    }
  }
  
  /// This vector's scalar values in a standard basis.
  public var scalars: Scalars {
    get { .init(base: self) }
    set { self = newValue.base  }
  }
}

extension Vector11: FixedSizeVector {
  public static var dimension: Int { return 11 }
}


/// An element of R^12, with Euclidean inner product.
public struct Vector12: Codable, KeyPathIterable {
  @differentiable(reverse) public var s0: Double
  @differentiable(reverse) public var s1: Double
  @differentiable(reverse) public var s2: Double
  @differentiable(reverse) public var s3: Double
  @differentiable(reverse) public var s4: Double
  @differentiable(reverse) public var s5: Double
  @differentiable(reverse) public var s6: Double
  @differentiable(reverse) public var s7: Double
  @differentiable(reverse) public var s8: Double
  @differentiable(reverse) public var s9: Double
  @differentiable(reverse) public var s10: Double
  @differentiable(reverse) public var s11: Double

  @differentiable(reverse)
  public init(_ s0: Double, _ s1: Double, _ s2: Double, _ s3: Double, _ s4: Double, _ s5: Double, _ s6: Double, _ s7: Double, _ s8: Double, _ s9: Double, _ s10: Double, _ s11: Double) {
    self.s0 = s0
    self.s1 = s1
    self.s2 = s2
    self.s3 = s3
    self.s4 = s4
    self.s5 = s5
    self.s6 = s6
    self.s7 = s7
    self.s8 = s8
    self.s9 = s9
    self.s10 = s10
    self.s11 = s11
  }
}

/// Conformance to Vector
extension Vector12: AdditiveArithmetic, Vector {
  @differentiable(reverse)
  public static func += (_ lhs: inout Self, _ rhs: Self) {
    lhs.s0 += rhs.s0
    lhs.s1 += rhs.s1
    lhs.s2 += rhs.s2
    lhs.s3 += rhs.s3
    lhs.s4 += rhs.s4
    lhs.s5 += rhs.s5
    lhs.s6 += rhs.s6
    lhs.s7 += rhs.s7
    lhs.s8 += rhs.s8
    lhs.s9 += rhs.s9
    lhs.s10 += rhs.s10
    lhs.s11 += rhs.s11
  }

  @differentiable(reverse)
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
    lhs.s0 -= rhs.s0
    lhs.s1 -= rhs.s1
    lhs.s2 -= rhs.s2
    lhs.s3 -= rhs.s3
    lhs.s4 -= rhs.s4
    lhs.s5 -= rhs.s5
    lhs.s6 -= rhs.s6
    lhs.s7 -= rhs.s7
    lhs.s8 -= rhs.s8
    lhs.s9 -= rhs.s9
    lhs.s10 -= rhs.s10
    lhs.s11 -= rhs.s11
  }

  @differentiable(reverse)
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
    lhs.s0 *= rhs
    lhs.s1 *= rhs
    lhs.s2 *= rhs
    lhs.s3 *= rhs
    lhs.s4 *= rhs
    lhs.s5 *= rhs
    lhs.s6 *= rhs
    lhs.s7 *= rhs
    lhs.s8 *= rhs
    lhs.s9 *= rhs
    lhs.s10 *= rhs
    lhs.s11 *= rhs
  }

  @differentiable(reverse)
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
    result += self.s0 * other.s0
    result += self.s1 * other.s1
    result += self.s2 * other.s2
    result += self.s3 * other.s3
    result += self.s4 * other.s4
    result += self.s5 * other.s5
    result += self.s6 * other.s6
    result += self.s7 * other.s7
    result += self.s8 * other.s8
    result += self.s9 * other.s9
    result += self.s10 * other.s10
    result += self.s11 * other.s11
    return result
  }

  public var dimension: Int { return 12 }

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    var index = scalars.startIndex
    self.s0 = scalars[index]
    index = scalars.index(after: index)
    self.s1 = scalars[index]
    index = scalars.index(after: index)
    self.s2 = scalars[index]
    index = scalars.index(after: index)
    self.s3 = scalars[index]
    index = scalars.index(after: index)
    self.s4 = scalars[index]
    index = scalars.index(after: index)
    self.s5 = scalars[index]
    index = scalars.index(after: index)
    self.s6 = scalars[index]
    index = scalars.index(after: index)
    self.s7 = scalars[index]
    index = scalars.index(after: index)
    self.s8 = scalars[index]
    index = scalars.index(after: index)
    self.s9 = scalars[index]
    index = scalars.index(after: index)
    self.s10 = scalars[index]
    index = scalars.index(after: index)
    self.s11 = scalars[index]
    index = scalars.index(after: index)
  }

  /// A type that can represent all of this vector's scalar values in a standard basis.
  public struct Scalars: RandomAccessCollection, MutableCollection {
    // Deduction of Indices fails without an explicit declaration.
    /// A type that can represent all the indices of elements in this collection.
    public typealias Indices = Range<Int>

    /// The vector whose scalars are reflected by `self`.
    internal var base: Vector12

    /// The position of the first element, or `endIndex` if `self.isEmpty`.
    public var startIndex: Int { 0 }
    
    /// The position one step beyond the last contained element.
    public var endIndex: Int { base.dimension }

    /// Accesses the scalar at `i`.
    public subscript(i: Int) -> Double {
      get {
        precondition(i >= 0 && i < endIndex)
        return withUnsafePointer(to: self) {
          UnsafeRawPointer($0).assumingMemoryBound(to: Double.self)[i]
        }
      }
      _modify {
        precondition(i >= 0 && i < endIndex)
        let p = withUnsafeMutablePointer(to: &self) { $0 }
        let q = UnsafeMutableRawPointer(p).assumingMemoryBound(to: Double.self)
        defer { _fixLifetime(self) }
        yield &q[i]
      }
    }
  }
  
  /// This vector's scalar values in a standard basis.
  public var scalars: Scalars {
    get { .init(base: self) }
    set { self = newValue.base  }
  }
}

extension Vector12: FixedSizeVector {
  public static var dimension: Int { return 12 }
}

