/// A fixed-size array of `Vector`s is a `Vector`.
extension FixedSizeArray where Element: Vector {
  public typealias Scalar = Element.Scalar
  public mutating func add(_ other: Self) {
    for i in indices {
      self[i].add(other[i])
    }
  }
  public mutating func scale(by scalar: Element.Scalar) {
    for i in indices {
      self[i].scale(by: scalar)
    }
  }
  public func bracket<Covector: Collection & Vector>(_ covector: Covector) -> Scalar
  where Self: Vector, Element.Covector == Covector.Element, Covector.Index == Index {
    var result: Scalar = 0
    var selfIterator = makeIterator()
    var covectorIterator = covector.makeIterator()
    while let selfElement = selfIterator.next() {
      result += selfElement.bracket(covectorIterator.next()!)
    }
    assert(covectorIterator.next() == nil)
    return result
  }
  public static var zero: Self {
    return Self((0..<Self.count).lazy.map { _ in Element.zero })
  }
}

// MARK: - "Generated Code"

extension Array1: AdditiveArithmetic, Vector where Element: Vector {
  public typealias Covector = Array1<Element.Covector>
}
extension Array2: AdditiveArithmetic, Vector where Element: Vector {
  public typealias Covector = Array2<Element.Covector>
}
extension Array3: AdditiveArithmetic, Vector where Element: Vector {
  public typealias Covector = Array3<Element.Covector>
}

extension Array1: Differentiable, DifferentiableVector where Element: DifferentiableVector {
  public typealias TangentVector = Covector
}
extension Array2: Differentiable, DifferentiableVector where Element: DifferentiableVector {
  public typealias TangentVector = Covector
}
extension Array3: Differentiable, DifferentiableVector where Element: DifferentiableVector {
  public typealias TangentVector = Covector
}

extension Array1: LinearMap where Element: Vector {}
extension Array2: LinearMap where Element: Vector {}
extension Array3: LinearMap where Element: Vector {}

public typealias Vector1<Scalar> = Array1<ScalarVector<Scalar>>
public typealias Vector2<Scalar> = Array2<ScalarVector<Scalar>>
public typealias Vector3<Scalar> = Array3<ScalarVector<Scalar>>

public typealias Vector1d = Vector1<Double>
public typealias Vector2d = Vector2<Double>
public typealias Vector3d = Vector3<Double>
