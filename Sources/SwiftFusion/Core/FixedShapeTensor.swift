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

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    self.init(Tensor(shape: Self.shape, scalars: Array(scalars)))
  }

  public var dimension: Int {
    Self.dimension
  }

  public static var dimension: Int {
    shape.reduce(1, *)
  }

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

  public static var zero: Self { .init(Tensor(zeros: Self.shape)) }
}

// Copy this implementation and modify the `shape` to create `FixedShapeTensor`s with other shapes.
/// A `Tensor` with shape `[10, 10]`.
public struct Tensor10x10: AdditiveArithmetic, FixedShapeTensor {
  public typealias TangentVector = Self
  public static var shape: TensorShape { [10, 10] }

  @differentiable public var tensor: Tensor<Double> {
    get { scalars.storage }
    set { scalars.storage = newValue }
  }

  /// A type that can represent all of this vector's scalar values in a standard basis.
  public struct Scalars
    : RandomAccessCollection, MutableCollection, Differentiable, AdditiveArithmetic
  {
    fileprivate var storage: Tensor<Double>

    /// The position of the first element, or `endIndex` if `self.isEmpty`.
    public var startIndex: Int { 0 }

    /// The position one step beyond the last contained element.
    public var endIndex: Int { 100 }

    /// Accesses the scalar at `i`.
    public subscript(i: Int) -> Double {
      get { storage[i / 10, i % 10].scalarized() }
      set { storage[i / 10, i % 10] = Tensor(newValue) }
    }
  }

  /// This vector's scalar values in a standard basis.
  @differentiable public var scalars: Scalars

  @differentiable
  public init(_ tensor: Tensor<Double>) {
    precondition(tensor.shape == Self.shape)
    self.scalars = Scalars(storage: tensor)
  }
}

/// A `Tensor` with shape `[28, 62, 1]`.
public struct Tensor28x62x1: AdditiveArithmetic, FixedShapeTensor {
  public typealias TangentVector = Self
  public static var shape: TensorShape { [28, 62, 1] }

  @differentiable public var tensor: Tensor<Double> {
    get { scalars.storage }
    set { scalars.storage = newValue }
  }

  /// A type that can represent all of this vector's scalar values in a standard basis.
  public struct Scalars
    : RandomAccessCollection, MutableCollection, Differentiable, AdditiveArithmetic
  {
    fileprivate var storage: Tensor<Double>

    /// The position of the first element, or `endIndex` if `self.isEmpty`.
    public var startIndex: Int { 0 }

    /// The position one step beyond the last contained element.
    public var endIndex: Int { 28 * 62 * 1 }

    /// Accesses the scalar at `i`.
    public subscript(i: Int) -> Double {
      get { storage[i / 62, i % 62, 0].scalarized() }
      set { storage[i / 62, i % 62, 0] = Tensor(newValue) }
    }
  }

  /// This vector's scalar values in a standard basis.
  @differentiable public var scalars: Scalars

  @differentiable
  public init(_ tensor: Tensor<Double>) {
    precondition(tensor.shape == Self.shape)
    self.scalars = Scalars(storage: tensor)
  }
}

/// A `Tensor` with shape `[28, 62, 3]`.
public struct Tensor28x62x3: AdditiveArithmetic, FixedShapeTensor {
  public typealias TangentVector = Self
  public static var shape: TensorShape { [28, 62, 3] }

  @differentiable public var tensor: Tensor<Double> {
    get { scalars.storage }
    set { scalars.storage = newValue }
  }

  /// A type that can represent all of this vector's scalar values in a standard basis.
  public struct Scalars
    : RandomAccessCollection, MutableCollection, Differentiable, AdditiveArithmetic
  {
    fileprivate var storage: Tensor<Double>

    /// The position of the first element, or `endIndex` if `self.isEmpty`.
    public var startIndex: Int { 0 }

    /// The position one step beyond the last contained element.
    public var endIndex: Int { 28 * 62 * 3 }

    /// Accesses the scalar at `i`.
    public subscript(i: Int) -> Double {
      get { storage[i / (62 * 3), (i / 3) % 62, i % 3].scalarized() }
      set { storage[i / (62 * 3), (i / 3) % 62, i % 3] = Tensor(newValue) }
    }
  }

  /// This vector's scalar values in a standard basis.
  @differentiable public var scalars: Scalars

  @differentiable
  public init(_ tensor: Tensor<Double>) {
    precondition(tensor.shape == Self.shape)
    self.scalars = Scalars(storage: tensor)
  }
}
