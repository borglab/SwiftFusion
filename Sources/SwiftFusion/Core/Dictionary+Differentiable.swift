import _Differentiation
/// This file makes `Dictionary` differentiable.
///
/// Note: This will eventually be moved into the Swift standard library. Once it is in the
/// standard library, we can delete it from this repository.

/// Implements the `Differentiable` requirements.
extension Dictionary: Differentiable where Value: Differentiable {
  public typealias TangentVector = Dictionary<Key, Value.TangentVector>
  public mutating func move(along direction: TangentVector) {
    for (componentKey, componentDirection) in direction {
      func fatalMissingComponent() -> Value {
        fatalError("missing component \(componentKey) in moved Dictionary")
      }
      self[componentKey, default: fatalMissingComponent()].move(along: componentDirection)
    }
  }

  public var zeroTangentVectorInitializer: () -> TangentVector {
    { mapValues { v in v.zeroTangentVector } }
  }
}

/// Implements the `AdditiveArithmetic` requirements.
extension Dictionary: AdditiveArithmetic where Value: AdditiveArithmetic {
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    lhs.merging(rhs, uniquingKeysWith: +)
  }
  public static func - (_ lhs: Self, _ rhs: Self) -> Self {
    lhs.merging(rhs.mapValues { .zero - $0 }, uniquingKeysWith: +)
  }
  public static var zero: Self { [:] }
}

/// Provides some differentiable methods for manipulating `Dictionary`.
///
/// Note: Once the differentiable `Dictionary` is moved into the standard library, the standard
/// `Dictionary` methods will be differentiable and you won't have to use special differentiable
/// methods to manipulate `Dictionary`.
extension Dictionary where Value: Differentiable {
  /// Returns the value with `key`.
  ///
  /// Precondition: `self` contains an entry with key `key`.
  @differentiable
  public func differentiableSubscript(_ key: Key) -> Value {
    self[key]!
  }

  @derivative(of: differentiableSubscript)
  @usableFromInline
  func vjpDifferentiableSubscript(_ key: Key)
    -> (value: Value, pullback: (Value.TangentVector) -> TangentVector)
  {
    (differentiableSubscript(key), { [key: $0] })
  }
}
