import Foundation
import TensorFlow

/// A vector in a Euclidean vector space.
public protocol EuclideanVector: Differentiable where Self.TangentVector == Self {
  // Note: This is a work in progress. We intend to add more requirements here as we need them.

  // MARK: - AdditiveArithmetic requirements, refined to require differentiability.

  @differentiable
  static func += (_ lhs: inout Self, _ rhs: Self)

  @differentiable
  static func + (_ lhs: Self, _ rhs: Self) -> Self

  @differentiable
  static func -= (_ lhs: inout Self, _ rhs: Self)

  @differentiable
  static func - (_ lhs: Self, _ rhs: Self) -> Self

  // MARK: - Scalar multiplication.

  @differentiable
  static func *= (_ lhs: inout Self, _ rhs: Double)

  @differentiable
  static func * (_ lhs: Double, _ rhs: Self) -> Self

  // MARK: - Euclidean structure.

  /// The inner product of `self` with `other`.
  @differentiable
  func dot(_ other: Self) -> Double

  // MARK: - Conversion to/from a concrete dynamically-sized `Vector`.

  /// Creates an instance with the same elements as `vector`.
  @differentiable
  init(_ vector: Vector)

  /// A `Vector` containint the same elements as `self`.
  @differentiable
  var vector: Vector { get }
}

/// Convenient operations on Euclidean vector spaces that can be implemented in terms of the
/// primitive operations.
extension EuclideanVector {
  @differentiable
  public static prefix func - (_ v: Self) -> Self {
    return (-1) * v
  }

  @differentiable
  public var squaredNorm: Double {
    return self.dot(self)
  }

  @differentiable
  public var norm: Double {
    return squaredNorm.squareRoot()
  }

  /// Returns this vector as a `Tensor<Double>`.
  @differentiable
  public var tensor: Tensor<Double> {
    let scalars = self.vector.scalars
    return Tensor<Double>(shape: [withoutDerivative(at: scalars).count], scalars: scalars)
  }

  /// Creates a vector with the same elements as `tensor`.
  @differentiable
  public init(_ tensor: Tensor<Double>) {
    let vector = Vector(tensor.scalars)
    self.init(vector)
  }
}

/// Default implementations of some `EuclideanVector` requirements.
extension EuclideanVector {
  @differentiable
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    var result = lhs
    result += rhs
    return result
  }

  @differentiable
  public static func - (_ lhs: Self, _ rhs: Self) -> Self {
    var result = lhs
    result -= rhs
    return result
  }

  @differentiable
  public static func * (_ lhs: Double, _ rhs: Self) -> Self {
    var result = rhs
    result *= lhs
    return result
  }
}

/// An Array is a `EuclideanVector` when its elements are `EuclideanVectorN`.
extension Array.DifferentiableView: EuclideanVector where Element: EuclideanVectorN {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
    // If `lhs` or `rhs` is empty, then it may be a "zero tangent vector" created by AD, and we
    // should treat it as a vector of zeros.
    guard rhs.base.count > 0 else { return }
    guard lhs.base.count > 0 else {
      lhs = rhs
      return
    }

    for index in lhs.base.indices {
      lhs.base[index] += rhs.base[index]
    }
  }

  @derivative(of: +=)
  @usableFromInline
  static func vjpAddSelf(_ lhs: inout Self, _ rhs: Self) -> (value: (), pullback: (inout Self) -> Self) {
    lhs += rhs
    func pullback(_ v: inout Self) -> Self {
      return v
    }
    return ((), pullback)
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
    // If `lhs` or `rhs` is empty, then it may be a "zero tangent vector" created by AD, and we
    // should treat it as a vector of zeros.
    guard rhs.base.count > 0 else { return }
    guard lhs.base.count > 0 else {
      lhs = -rhs
      return
    }

    for index in lhs.base.indices {
      lhs.base[index] -= rhs.base[index]
    }
  }

  @derivative(of: -=)
  @usableFromInline
  static func vjpSubtractSelf(_ lhs: inout Self, _ rhs: Self) -> (value: (), pullback: (inout Self) -> Self) {
    lhs -= rhs
    func pullback(_ v: inout Self) -> Self {
      return -v
    }
    return ((), pullback)
  }


  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
    for index in lhs.base.indices {
      lhs.base[index] *= rhs
    }
  }

  @derivative(of: *=)
  @usableFromInline
  static func vjpMultiplyScalar(_ lhs: inout Self, _ rhs: Double) -> (value: (), pullback: (inout Self) -> Double) {
    var pullbacks: [(Element) -> (Double, Element)] = []
    pullbacks.reserveCapacity(lhs.base.count)
    for index in lhs.base.indices {
      let (value, pb) = valueWithPullback(at: rhs, lhs.base[index], in: *)
      lhs.base[index] = value
      pullbacks.append(pb)
    }
    func pullback(_ v: inout Self) -> Double {
      var tRhs = Double(0)
      for index in v.indices {
        let (wrtScalar, wrtElement) = pullbacks[index](v[index])
        v[index] = wrtElement
        tRhs += wrtScalar
      }
      return tRhs
    }
    return ((), pullback)
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    var result = Double(0)
    for index in base.indices {
      result += base[index].dot(other.base[index])
    }
    return result
  }

  @derivative(of: dot)
  @usableFromInline
  func vjpDot(_ other: Self) -> (value: Double, pullback: (Double) -> (Self, Self)) {
    return (self.dot(other), { v in (v * other, v * self) })
  }

  @differentiable
  public init(_ vector: Vector) {
    let elements = (0..<(vector.scalars.count / Element.dimension)).map { index -> Element in
      let scalars = vector.scalars[(Element.dimension * index)..<(Element.dimension * (index + 1))]
      return Element(Vector(Array<Double>(scalars)))
    }
    self.init(elements)
  }

  @derivative(of: init)
  @usableFromInline
  static func vjpVectorInit(_ vector: Vector) ->
    (value: Self, pullback: (TangentVector) -> Vector)
  {
    fatalError("not implemented")
  }

  @differentiable
  public var vector: Vector {
    let scalars = base.flatMap { $0.vector.scalars }
    return Vector(scalars)
  }

  @derivative(of: vector)
  @usableFromInline
  func vjpVector() -> (value: Vector, pullback: (Vector) -> TangentVector) {
    fatalError("not implemented")
  }
}
