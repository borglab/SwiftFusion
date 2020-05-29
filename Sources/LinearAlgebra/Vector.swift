/// Views `Scalars` as a vector.
public struct Vector<Scalars: Equatable & FixedSizeArray>: VectorProtocol
where Scalars.Element: Numeric {
  /// The elements of the vector.
  public var scalars: Scalars
  
  /// Creates a vector containing `scalars`.
  public init(_ scalars: Scalars) {
    self.scalars = scalars
  }
  
  // MARK: - `VectorProtocol` conformance.
  
  public typealias Scalar = Scalars.Element
  public typealias Covector = Self
  public mutating func add(_ other: Self) {
    for index in scalars.indices {
      scalars[index] += other.scalars[index]
    }
  }
  public mutating func scale(by scalar: Scalar) {
    for index in scalars.indices {
      scalars[index] *= scalar
    }
  }
  public func bracket(_ covector: Covector) -> Scalar {
    var result: Scalar = 0
    var scalarsIterator = scalars.makeIterator()
    var covectorScalarsIterator = covector.scalars.makeIterator()
    while let scalar = scalarsIterator.next() {
      result += scalar * covectorScalarsIterator.next()!
    }
    assert(covectorScalarsIterator.next() == nil)
    return result
  }
  public static var zero: Self {
    return Self(Scalars((0..<Scalars.count).lazy.map { _ in Scalar.zero }))
  }
}

extension Vector: Equatable where Scalars: Equatable {}

extension Vector: Differentiable, DifferentiableVector
where Scalar: Differentiable, Scalar.TangentVector == Scalar {
  public typealias TangentVector = Self
}

// MARK: - "Generated Code"

typealias Vector1 = Vector<Array1<Double>>
typealias Vector2 = Vector<Array2<Double>>
typealias Vector3 = Vector<Array3<Double>>

extension Vector {
  public init(_ s0: Double) where Scalars == Array1<Double> {
    self.scalars = Array1(s0)
  }
  public init(_ s0: Double, _ s1: Double) where Scalars == Array2<Double> {
    self.scalars = Array2(s0, s1)
  }
  public init(_ s0: Double, _ s1: Double, _ s2: Double) where Scalars == Array3<Double> {
    self.scalars = Array3(s0, s1, s2)
  }
}
