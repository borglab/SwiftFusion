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
  @differentiable
  public func sum() -> Double {
    return scalars.differentiableReduce(0, +)
  }
}

/// EuclideanVector conformance.
extension Vector: AdditiveArithmetic, EuclideanVector {
  @differentiable
  public static func += (_ lhs: inout Vector, _ rhs: Vector) {
    // If `lhs` or `rhs` is empty, then it may be a "zero tangent vector" created by AD, and we
    // should treat it as a vector of zeros.
    guard rhs.scalarsStorage.count > 0 else { return }
    guard lhs.scalarsStorage.count > 0 else {
      lhs = rhs
      return
    }

    for index in lhs.scalarsStorage.indices {
      lhs.scalarsStorage[index] += rhs.scalarsStorage[index]
    }
  }

  @derivative(of: +=)
  @usableFromInline
  static func vjpAddVector(_ lhs: inout Vector, _ rhs: Vector) -> (value: (), pullback: (inout Vector) -> Vector) {
    lhs += rhs
    func pullback(_ v: inout Vector) -> Vector {
      return v
    }
    return ((), pullback)
  }

  @differentiable
  public static func -= (_ lhs: inout Vector, _ rhs: Vector) {
    // If `lhs` or `rhs` is empty, then it may be a "zero tangent vector" created by AD, and we
    // should treat it as a vector of zeros.
    guard rhs.scalarsStorage.count > 0 else { return }
    guard lhs.scalarsStorage.count > 0 else {
      lhs = -rhs
      return
    }

    for index in lhs.scalarsStorage.indices {
      lhs.scalarsStorage[index] -= rhs.scalarsStorage[index]
    }
  }

  @derivative(of: -=)
  @usableFromInline
  static func vjpSubtractVector(_ lhs: inout Vector, _ rhs: Vector) -> (value: (), pullback: (inout Vector) -> Vector) {
    lhs -= rhs
    func pullback(_ v: inout Vector) -> Vector {
      return -v
    }
    return ((), pullback)
  }

  /// The zero vector.
  ///
  /// Note: "Zero" doesn't make very much sense as a static property on a dynamically sized vector
  /// because we don't known the dimension of the zero. However, `AdditiveArithmetic` requires it,
  /// so we implement it as reasonably as possible.
  public static var zero: Vector {
    return Vector([])
  }

  @differentiable
  public static func *= (_ lhs: inout Vector, _ rhs: Double) {
    for index in lhs.scalarsStorage.indices {
      lhs.scalarsStorage[index] *= rhs
    }
  }

  @derivative(of: *=)
  @usableFromInline
  static func vjpMultiplyScalar(_ lhs: inout Vector, _ rhs: Double) -> (value: (), pullback: (inout Vector) -> Double) {
    let originalLhs = lhs
    lhs *= rhs
    func pullback(_ v: inout Vector) -> Double {
      let tRhs = originalLhs.dot(v)
      v *= rhs
      return tRhs
    }
    return ((), pullback)
  }

  @differentiable
  public func dot(_ other: Vector) -> Double {
    var result = Double(0)
    for index in scalarsStorage.indices {
      result += self.scalarsStorage[index] * other.scalarsStorage[index]
    }
    return result
  }

  @derivative(of: dot)
  @usableFromInline
  func vjpDot(_ other: Vector) -> (value: Double, pullback: (Double) -> (Vector, Vector)) {
    return (self.dot(other), { v in (v * other, v * self) })
  }
}

/// Conversion to tensor.
extension Vector {
  /// Returns this vector as a `Tensor<Double>`.
  public var tensor: Tensor<Double> {
    return Tensor(shape: [scalarsStorage.count], scalars: scalarsStorage)
  }
}
