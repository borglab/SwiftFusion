import TensorFlow

/// A dynamically sized vector.
public struct Vector: Equatable, Differentiable {
  public typealias TangentVector = Self

  /// The scalars in the vector.
  @differentiable public var scalars: [Double] { return scalarsStorage }

  /// Derivative for `scalars`.
  // This is necessary because we have specified a custom `TangentVector`.
  @derivative(of: scalars)
  @usableFromInline
  func vjpScalars()
    -> (value: [Double], pullback: (Array<Double>.DifferentiableView) -> TangentVector)
  {
    return (scalars, { Vector($0.base) })
  }

  /// The storage for the scalars.
  // This works around the fact that we cannot define custom derivatives for stored properties.
  // TODO: Once we can define custom derivatives for stored properties, remove this.
  internal var scalarsStorage: [Double]
}

// TODO: Conform vector to the appropriate collection protocols instead of defining these manually.
extension Vector {
  mutating func replaceSubrange<C>(_ subrange: Range<Int>, with newElements: C)
    where C: Collection, C.Element == Double
  {
    self.scalarsStorage.replaceSubrange(subrange, with: newElements)
  }
}

/// Can be losslessly converted to and from a `Vector`.
public protocol VectorConvertible: Differentiable {
  @differentiable
  init(_ vector: Vector)

  @differentiable
  var vector: Vector { get }
}

/// Initializers.
extension Vector {
  /// Creates a vector with the given `scalars`.
  @differentiable
  public init(_ scalars: [Double]) {
    self.scalarsStorage = scalars
  }

  /// Derivative for `init`.
  // This is necessary because we have specified a custom `TangentVector`.
  @derivative(of: init(_:))
  @usableFromInline
  static func vjpInit(_ scalars: [Double])
    -> (value: Vector, pullback: (TangentVector) -> Array<Double>.DifferentiableView)
  {
    return (Vector(scalars), { Array<Double>.DifferentiableView($0.scalars) })
  }

  /// Creates a zero vector of the given `dimension`.
  public init(zeros dimension: Int) {
    self.init(Array(repeating: 0, count: dimension))
  }
}

/// Miscellaneous computed properties.
extension Vector {
  public var dimension: Int {
    return scalarsStorage.count
  }
}

/// Arithmetic on elements.
extension Vector {
  /// Sum of the elements of `self`.
  public func sum() -> Double {
    return scalarsStorage.reduce(0, +)
  }

  /// Vector whose elements are squares of the elements of `self`.
  public func squared() -> Self {
    return Vector(scalarsStorage.map { $0 * $0 })
  }
}

/// Euclidean norms.
extension Vector {
  /// Euclidean norm of `self`.
  public var norm: Double { squaredNorm.squareRoot() }

  /// Square of the Euclidean norm of `self`.
  @differentiable
  public var squaredNorm: Double { self.squared().sum() }

  /// Derivative of `squaredNorm`.
  // TODO: This is a custom derivative because derivatives of `map` and
  // `reduce` are currently very slow. We can use the automatic derivative for this once
  // https://github.com/apple/swift/pull/31704 is available.
  @derivative(of: squaredNorm)
  @usableFromInline
  func vjpSquaredNorm() -> (value: Double, pullback: (Double) -> TangentVector) {
    return (
      value: squaredNorm,
      pullback: { self.scaled(by: 2 * $0) }
    )
  }
}

/// AdditiveArithmetic conformance.
extension Vector: AdditiveArithmetic {
  public static func += (_ lhs: inout Vector, _ rhs: Vector) {
    for index in lhs.scalarsStorage.indices {
      lhs.scalarsStorage[index] += rhs.scalarsStorage[index]
    }
  }

  public static func + (_ lhs: Vector, _ rhs: Vector) -> Vector {
    var result = lhs
    result += rhs
    return result
  }

  public static func -= (_ lhs: inout Vector, _ rhs: Vector) {
    for index in lhs.scalarsStorage.indices {
      lhs.scalarsStorage[index] -= rhs.scalarsStorage[index]
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
    for index in scalarsStorage.indices {
      scalarsStorage[index] += x
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
    for index in scalarsStorage.indices {
      scalarsStorage[index] -= x
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
    for index in scalarsStorage.indices {
      scalarsStorage[index] *= scalar
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

extension Vector: EuclideanVectorSpace {}

/// Conversion to tensor.
extension Vector {
  /// Returns this vector as a `Tensor<Double>`.
  public var tensor: Tensor<Double> {
    return Tensor(shape: [scalarsStorage.count], scalars: scalarsStorage)
  }
}
