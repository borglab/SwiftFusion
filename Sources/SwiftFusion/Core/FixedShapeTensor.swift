import TensorFlow

/// A `Tensor` with a statically known shape.
///
/// TODO(https://github.com/borglab/SwiftFusion/issues/152): Remove this.
public protocol FixedShapeTensor: FixedSizeVector {
  /// The shape of an instance.
  static var shape: TensorShape { get }

  /// Creates an instance containing `tensor`.
  ///
  /// - Requires: `tensor.shape == Self.shape`.
  @differentiable init(_ tensor: Tensor<Double>)

  /// The value.
  @differentiable var tensor: Tensor<Double> { get set }
}

/// Default implementations of `Vector` requirements.
extension FixedShapeTensor {
  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
    lhs = Self(lhs.tensor + rhs.tensor)
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
    lhs = Self(lhs.tensor - rhs.tensor)
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
    lhs = Self(lhs.tensor * rhs)
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    (self.tensor * other.tensor).sum().scalarized()
  }

  public var dimension: Int {
    Self.dimension
  }

  public static var dimension: Int {
    shape.reduce(1, *)
  }

  public static var zero: Self { Self(Tensor(zeros: shape)) }

  public var standardBasis: [Self] {
    (0..<dimension).map { i in
      var b = Array(repeating: Double(0), count: dimension)
      b[i] = 1
      return Self(Tensor(shape: Self.shape, scalars: b))
    }
  }

  /// Returns the result of calling `body` on the scalars of `self`.
  public func withUnsafeBufferPointer<R>(
    _ body: (UnsafeBufferPointer<Double>) throws -> R
  ) rethrows -> R {
    try self.tensor.scalars.withUnsafeBufferPointer(body)
  }

  /// Returns the result of calling `body` on the scalars of `self`.
  public mutating func withUnsafeMutableBufferPointer<R>(
    _ body: (UnsafeMutableBufferPointer<Double>) throws -> R
  ) rethrows -> R {
    var scalars = self.tensor.scalars
    let r = try scalars.withUnsafeMutableBufferPointer { b in
      try body(b)
    }
    self.tensor = Tensor(shape: Self.shape, scalars: scalars)
    return r
  }
}

// Copy this implementation and modify the `shape` to create `FixedShapeTensor`s with other shapes.
/// A `Tensor` with shape `[10, 10]`.
public struct Tensor10x10: AdditiveArithmetic, FixedShapeTensor {
  public typealias TangentVector = Tensor10x10
  public static var shape: TensorShape { [10, 10] }
  @differentiable public var tensor: Tensor<Double>

  @differentiable
  public init(_ tensor: Tensor<Double>) {
    precondition(tensor.shape == Self.shape)
    self.tensor = tensor
  }
}
