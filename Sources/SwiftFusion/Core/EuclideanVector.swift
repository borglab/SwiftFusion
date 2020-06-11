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

  // MARK: - Conversion to/from collections of scalars.

  /// Creates an instance whose elements are `scalars`.
  ///
  /// Precondition: `scalars` must have an element count that `Self` can hold (e.g. if `Self` is a
  /// fixed-size vectors, then `scalars` must have exactly the right number of elements).
  ///
  /// TODO: Maybe make this failable.
  init<Source: Collection>(_ scalars: Source) where Source.Element == Double

  /// The scalars in `self`.
  var scalars: [Double] { get }
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

  /// Returns this vector as a `Vector`.
  ///
  /// Note: This is for backwards compatibility with existing code.
  public var vector: Vector {
    return Vector(Array(scalars))
  }

  /// Creates a vector with the same elements as `vector`.
  ///
  /// Note: This is for backwards compatibility with existing code.
  public init(_ vector: Vector) {
    self.init(vector.scalars)
  }

  /// Returns this vector as a `Tensor<Double>`.
  public var tensor: Tensor<Double> {
    let scalars = self.vector.scalars
    return Tensor<Double>(shape: [withoutDerivative(at: scalars).count], scalars: scalars)
  }

  /// Creates a vector with the same elements as `tensor`.
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

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    // No one currently uses this and we intend to remove its soon.
    fatalError("not implemented")
  }

  public var scalars: [Double] {
    // No one currently uses this and we intend to remove its soon.
    fatalError("not implemented")
  }
}
