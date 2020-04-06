import TensorFlow

/// A type that is convertible to/from a `Tensor`.
public protocol TensorConvertible: Differentiable {
  /// Create `Self` from a `Tensor` of shape `[dimension]`.
  @differentiable init(_ tensor: Tensor<Double>)

  /// A `Tensor` of shape `[dimension]`.
  var tensor: Tensor<Double> { @differentiable get }

  /// The dimension of this type.
  static var dimension: Int { get }
}

/// A pair of `TensorConvertible` types.
///
/// Useful for defining factors that take two values.
public struct TensorConvertiblePair<A: TensorConvertible, B: TensorConvertible>: TensorConvertible {
  public var a: A
  public var b: B

  public init(_ a: A, _ b: B) {
    self.a = a
    self.b = b
  }

  @differentiable
  public init(_ tensor: Tensor<Double>) {
    self.a = A(tensor.slice(lowerBounds: [0], upperBounds: [A.dimension]))
    self.b = B(tensor.slice(lowerBounds: [A.dimension], upperBounds: [Self.dimension]))
  }

  @differentiable
  public var tensor: Tensor<Double> {
    Tensor(concatenating: [a.tensor, b.tensor])
  }

  public static var dimension: Int { A.dimension + B.dimension }

  public struct TangentVector: AdditiveArithmetic, Differentiable {
    public var a: A.TangentVector
    public var b: B.TangentVector

    public init(_ a: A.TangentVector, _ b: B.TangentVector) {
      self.a = a
      self.b = b
    }
  }
  public mutating func move(along direction: TangentVector) { fatalError() }
}

extension TensorConvertiblePair.TangentVector: TensorConvertible
  where A.TangentVector: TensorConvertible, B.TangentVector: TensorConvertible
{
  @differentiable
  public init(_ tensor: Tensor<Double>) {
    self.a = A.TangentVector(tensor.slice(lowerBounds: [0], upperBounds: [A.dimension]))
    self.b = B.TangentVector(tensor.slice(lowerBounds: [A.dimension], upperBounds: [Self.dimension]))
  }

  @differentiable
  public var tensor: Tensor<Double> {
    Tensor(concatenating: [a.tensor, b.tensor])
  }

  public static var dimension: Int { A.dimension + B.dimension }
}
