//===----------------------------------------------------------------------===//
// `AnyNonlinearFactor`
//===----------------------------------------------------------------------===//

internal protocol _AnyNonlinearFactorBox {
  typealias ScalarType = Double
  // `Differentiable` requirements.
  mutating func _move(along direction: AnyDerivative)

  /// The underlying base value, type-erased to `Any`.
  var _typeErasedBase: Any { get }

  /// Returns the underlying value unboxed to the given type, if possible.
  func _unboxed<U: NonlinearFactor>(to type: U.Type) -> U?
  
  var _keys: Array<Int> { get }
  
  @differentiable(wrt: values)
  func _error(_ values: Values) -> ScalarType
  
  func _linearize(_ values: Values) -> JacobianFactor
}

internal struct _ConcreteNonlinearFactorBox<T: NonlinearFactor>: _AnyNonlinearFactorBox
{
  var _keys: Array<Int> {
    get {
      _base.keys
    }
  }
  
  /// The underlying base value.
  var _base: T

  init(_ base: T) {
    self._base = base
  }

  /// The underlying base value, type-erased to `Any`.
  var _typeErasedBase: Any {
    return _base
  }

  func _unboxed<U: NonlinearFactor>(to type: U.Type) -> U? {
    return (self as? _ConcreteNonlinearFactorBox<U>)?._base
  }

  mutating func _move(along direction: AnyDerivative) {
    guard
      let directionBase =
        direction.base as? T.TangentVector
    else {
      _derivativeTypeMismatch(T.self, type(of: direction.base))
    }
    _base.move(along: directionBase)
  }
  
  @differentiable(wrt: values)
  func _error(_ values: Values) -> ScalarType {
    _base.error(values)
  }
  
  func _linearize(_ values: Values) -> JacobianFactor {
    _base.linearize(values)
  }
}

public struct AnyNonlinearFactor: NonlinearFactor {
  internal var _box: _AnyNonlinearFactorBox

  internal init(_box: _AnyNonlinearFactorBox) {
    self._box = _box
  }

  /// The underlying base value.
  public var base: Any {
    return _box._typeErasedBase
  }

  /// Creates a type-erased derivative from the given derivative.
  @differentiable
  public init<T: NonlinearFactor>(_ base: T) {
    self._box = _ConcreteNonlinearFactorBox<T>(base)
  }

  @inlinable
  @derivative(of: init)
  internal static func _vjpInit<T: NonlinearFactor>(
    _ base: T
  ) -> (value: AnyNonlinearFactor, pullback: (AnyDerivative) -> T.TangentVector)
  {
    return (AnyNonlinearFactor(base), { v in v.base as! T.TangentVector })
  }

  @inlinable
  @derivative(of: init)
  internal static func _jvpInit<T: NonlinearFactor>(
    _ base: T
  ) -> (
    value: AnyNonlinearFactor, differential: (T.TangentVector) -> AnyDerivative
  ) {
    return (AnyNonlinearFactor(base), { dbase in AnyDerivative(dbase) })
  }

  public typealias TangentVector = AnyDerivative

  public mutating func move(along direction: TangentVector) {
    _box._move(along: direction)
  }
}

extension AnyNonlinearFactor {
  @differentiable
  public func baseAs<T: NonlinearFactor>(_ t: T.Type) -> T {
    base as! T
  }

  @derivative(of: baseAs)
  @usableFromInline
  func jvpBaseAs<T: NonlinearFactor>(_ t: T.Type) -> (
    value: T,
    differential: (AnyDerivative) -> T.TangentVector
  ) {
    (baseAs(t), { $0.base as! T.TangentVector })
  }

  @derivative(of: baseAs)
  @usableFromInline
  func vjpBaseAs<T: NonlinearFactor>(_ t: T.Type) -> (
    value: T,
    pullback: (T.TangentVector) -> AnyDerivative
  ) {
    (baseAs(t), { AnyDerivative($0) })
  }
}

extension AnyNonlinearFactor {
  @differentiable(wrt: values)
  public func error(_ values: Values) -> ScalarType {
    _box._error(values)
  }
  
  public var keys: Array<Int> {
    get {
      _box._keys
    }
  }
  
  public func linearize(_ values: Values) -> JacobianFactor {
    _box._linearize(values)
  }
}
