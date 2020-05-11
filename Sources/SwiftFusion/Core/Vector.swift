import TensorFlow

/// A dynamically sized vector.
public struct Vector: Equatable, Differentiable {
  /// The scalars in the vector.
  public var scalars: [Double]

  public typealias TangentVector = Self
}

/// Can be losslessly converted to and from a `Vector`.
public protocol VectorConvertible {
  init(_ vector: Vector)
  var vector: Vector { get }
}

/// Initializers.
extension Vector {
  /// Creates a vector with the given `scalars`.
  public init(_ scalars: [Double]) {
    self.scalars = scalars
  }

  /// Creates a zero vector of the given `dimension`.
  public init(zeros dimension: Int) {
    self.init(Array(repeating: 0, count: dimension))
  }
}

/// Miscellaneous computed properties.
extension Vector {
  public var dimension: Int {
    return scalars.count
  }
}

/// Arithmetic on elements.
extension Vector {
  /// Sum of the elements of `self`.
  public func sum() -> Double {
    return scalars.reduce(0, +)
  }

  /// Vector whose elements are squares of the elements of `self`.
  public func squared() -> Self {
    return Vector(scalars.map { $0 * $0 })
  }
}

/// Euclidean norms.
extension Vector {
  /// Euclidean norm of `self`.
  public var norm: Double { squaredNorm.squareRoot() }

  /// Square of the Euclidean norm of `self`.
  public var squaredNorm: Double { self.squared().sum() }
}

/// AdditiveArithmetic conformance.
extension Vector: AdditiveArithmetic {
  public static func += (_ lhs: inout Vector, _ rhs: Vector) {
    for index in lhs.scalars.indices {
      lhs.scalars[index] += rhs.scalars[index]
    }
  }

  public static func + (_ lhs: Vector, _ rhs: Vector) -> Vector {
    var result = lhs
    result += rhs
    return result
  }

  public static func -= (_ lhs: inout Vector, _ rhs: Vector) {
    for index in lhs.scalars.indices {
      lhs.scalars[index] -= rhs.scalars[index]
    }
  }

  public static func - (_ lhs: Vector, _ rhs: Vector) -> Vector {
    var result = lhs
    result -= rhs
    return result
  }

  /// The zero vector.
  ///
  /// Note: "Zero" doesn't make very much sense as a static property on a dynamically sized vector
  /// because we don't known the dimension of the zero. However, `AdditiveArithmetic` requires it,
  /// so we implement it as reasonably as possible.
  public static var zero: Vector {
    return Vector([])
  }
}

/// VectorProtocol conformance.
extension Vector: VectorProtocol {
  public typealias VectorSpaceScalar = Double

  public mutating func add(_ x: Double) {
    for index in scalars.indices {
      scalars[index] += x
    }
  }

  public func adding(_ x: Double) -> Vector {
    var result = self
    result.add(x)
    return result
  }

  public static func += (_ lhs: inout Vector, _ rhs: Double) {
    lhs.add(rhs)
  }

  public mutating func subtract(_ x: Double) {
    for index in scalars.indices {
      scalars[index] -= x
    }
  }

  public func subtracting(_ x: Double) -> Vector {
    var result = self
    result.subtract(x)
    return result
  }

  public static func -= (_ lhs: inout Vector, _ rhs: Double) {
    lhs.subtract(rhs)
  }

  public mutating func scale(by scalar: Double) {
    for index in scalars.indices {
      scalars[index] *= scalar
    }
  }

  public func scaled(by scalar: Double) -> Vector {
    var result = self
    result.scale(by: scalar)
    return result
  }

  public static func *= (_ lhs: inout Vector, _ rhs: Double) {
    lhs.scale(by: rhs)
  }

  public static func * (_ lhs: Double, _ rhs: Vector) -> Vector {
    return rhs.scaled(by: lhs)
  }
}

/// Conversion to tensor.
extension Vector {
  /// Returns this vector as a `Tensor<Double>`.
  public var tensor: Tensor<Double> {
    return Tensor(shape: [scalars.count], scalars: scalars)
  }
}
