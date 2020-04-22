//===----------------------------------------------------------------------===//
// `AnyNonlinearFactor`
//===----------------------------------------------------------------------===//

internal protocol _AnyNonlinearFactorBox {
  typealias ScalarType = Double

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
  public init<T: NonlinearFactor>(_ base: T) {
    self._box = _ConcreteNonlinearFactorBox<T>(base)
  }
}

extension AnyNonlinearFactor {
  public func baseAs<T: NonlinearFactor>(_ t: T.Type) -> T {
    base as! T
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
