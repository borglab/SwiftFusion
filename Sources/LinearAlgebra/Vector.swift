/// Views `Scalars` as a vector.
public struct Vector<Scalars: Equatable & FixedSizeArray>: MutableCollection, VectorProtocol
where Scalars.Element: Numeric {
  /// The elements of the vector.
  public var scalars: Scalars
  
  /// Creates a vector containing `scalars`.
  public init(_ scalars: Scalars) {
    self.scalars = scalars
  }
  
  // MARK: - `MutableCollection` conformance.
  
  public subscript(index: Scalars.Index) -> Scalars.Element {
    get {
      return self.scalars[index]
    }
    set(newValue) {
      self.scalars[index] = newValue
    }
  }

  public func index(after i: Scalars.Index) -> Scalars.Index { return scalars.index(after: i) }
  public var startIndex: Scalars.Index { scalars.startIndex }
  public var endIndex: Scalars.Index { scalars.endIndex }
  
  // MARK: - `VectorProtocol` conformance.
  
  public typealias Scalar = Scalars.Element
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
  public static var zero: Self {
    return Self(Scalars((0..<Scalars.count).lazy.map { _ in Scalar.zero }))
  }
}

extension Vector: Equatable where Scalars: Equatable {}

// MARK: - "Generated Code"

public typealias Vector1 = Vector<Array1<Double>>
public typealias Vector2 = Vector<Array2<Double>>
public typealias Vector3 = Vector<Array3<Double>>

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
