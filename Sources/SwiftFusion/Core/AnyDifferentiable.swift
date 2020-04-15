// NOTE: This will be available in S4TF v0.9, so we should delete this as soon as we switch to
// that version.

//===----------------------------------------------------------------------===//
// `AnyDifferentiable`
//===----------------------------------------------------------------------===//

internal protocol _AnyDifferentiableBox {
  // `Differentiable` requirements.
  mutating func _move(along direction: AnyDerivative)

  /// The underlying base value, type-erased to `Any`.
  var _typeErasedBase: Any { get }

  /// Returns the underlying value unboxed to the given type, if possible.
  func _unboxed<U: Differentiable>(to type: U.Type) -> U?
}

internal struct _ConcreteDifferentiableBox<T: Differentiable>: _AnyDifferentiableBox
{
  /// The underlying base value.
  var _base: T

  init(_ base: T) {
    self._base = base
  }

  /// The underlying base value, type-erased to `Any`.
  var _typeErasedBase: Any {
    return _base
  }

  func _unboxed<U: Differentiable>(to type: U.Type) -> U? {
    return (self as? _ConcreteDifferentiableBox<U>)?._base
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
}

public struct AnyDifferentiable: Differentiable {
  internal var _box: _AnyDifferentiableBox

  internal init(_box: _AnyDifferentiableBox) {
    self._box = _box
  }

  /// The underlying base value.
  public var base: Any {
    return _box._typeErasedBase
  }

  /// Creates a type-erased derivative from the given derivative.
  @differentiable
  public init<T: Differentiable>(_ base: T) {
    self._box = _ConcreteDifferentiableBox<T>(base)
  }

  @inlinable
  @derivative(of: init)
  internal static func _vjpInit<T: Differentiable>(
    _ base: T
  ) -> (value: AnyDifferentiable, pullback: (AnyDerivative) -> T.TangentVector)
  {
    return (AnyDifferentiable(base), { v in v.base as! T.TangentVector })
  }

  @inlinable
  @derivative(of: init)
  internal static func _jvpInit<T: Differentiable>(
    _ base: T
  ) -> (
    value: AnyDifferentiable, differential: (T.TangentVector) -> AnyDerivative
  ) {
    return (AnyDifferentiable(base), { dbase in AnyDerivative(dbase) })
  }

  public typealias TangentVector = AnyDerivative

  public mutating func move(along direction: TangentVector) {
    _box._move(along: direction)
  }
}

extension AnyDifferentiable {
  @differentiable
  public func baseAs<T: Differentiable>(_ t: T.Type) -> T {
    base as! T
  }

  @derivative(of: baseAs)
  @usableFromInline
  func jvpBaseAs<T: Differentiable>(_ t: T.Type) -> (
    value: T,
    differential: (AnyDerivative) -> T.TangentVector
  ) {
    (baseAs(t), { $0.base as! T.TangentVector })
  }

  @derivative(of: baseAs)
  @usableFromInline
  func vjpBaseAs<T: Differentiable>(_ t: T.Type) -> (
    value: T,
    pullback: (T.TangentVector) -> AnyDerivative
  ) {
    (baseAs(t), { AnyDerivative($0) })
  }
}

//===----------------------------------------------------------------------===//
// `AnyDerivative`
//===----------------------------------------------------------------------===//

@usableFromInline
internal protocol _AnyDerivativeBox {
  // `Equatable` requirements (implied by `AdditiveArithmetic`).
  func _isEqual(to other: _AnyDerivativeBox) -> Bool
  func _isNotEqual(to other: _AnyDerivativeBox) -> Bool

  // `AdditiveArithmetic` requirements.
  static var _zero: _AnyDerivativeBox { get }
  func _adding(_ x: _AnyDerivativeBox) -> _AnyDerivativeBox
  func _subtracting(_ x: _AnyDerivativeBox) -> _AnyDerivativeBox

  // `Differentiable` requirements.
  mutating func _move(along direction: _AnyDerivativeBox)

  /// The underlying base value, type-erased to `Any`.
  var _typeErasedBase: Any { get }

  /// Returns the underlying value unboxed to the given type, if possible.
  func _unboxed<U>(to type: U.Type) -> U?
  where U: Differentiable, U.TangentVector == U
}

extension _AnyDerivativeBox {
  /// Returns true if the underlying value has type `AnyDerivative.OpaqueZero`.
  @inlinable
  func _isOpaqueZero() -> Bool {
    return _unboxed(to: AnyDerivative.OpaqueZero.self) != nil
  }
}

@frozen
@usableFromInline
internal struct _ConcreteDerivativeBox<T>: _AnyDerivativeBox
where T: Differentiable, T.TangentVector == T {
  /// The underlying base value.
  @usableFromInline
  var _base: T

  @inlinable
  internal init(_ base: T) {
    self._base = base
  }

  /// The underlying base value, type-erased to `Any`.
  @inlinable
  var _typeErasedBase: Any {
    return _base
  }

  @inlinable
  func _unboxed<U>(to type: U.Type) -> U?
  where U: Differentiable, U.TangentVector == U {
    return (self as? _ConcreteDerivativeBox<U>)?._base
  }

  // `Equatable` requirements (implied by `AdditiveArithmetic`).
  @inlinable
  func _isEqual(to other: _AnyDerivativeBox) -> Bool {
    return _base == other._unboxed(to: T.self)
  }
  @inlinable
  func _isNotEqual(to other: _AnyDerivativeBox) -> Bool {
    return _base != other._unboxed(to: T.self)
  }

  // `AdditiveArithmetic` requirements.

  @inlinable
  static var _zero: _AnyDerivativeBox {
    return _ConcreteDerivativeBox(T.zero)
  }

  @inlinable
  func _adding(_ x: _AnyDerivativeBox) -> _AnyDerivativeBox {
    // 0 + x = x
    if _isOpaqueZero() {
      return x
    }
    // y + 0 = y
    if x._isOpaqueZero() {
      return self
    }
    guard let xBase = x._unboxed(to: T.self) else {
      _derivativeTypeMismatch(T.self, type(of: x._typeErasedBase))
    }
    return _ConcreteDerivativeBox(_base + xBase)
  }

  @inlinable
  func _subtracting(_ x: _AnyDerivativeBox) -> _AnyDerivativeBox {
    // y - 0 = y
    if x._isOpaqueZero() {
      return self
    }
    // 0 - x = -x
    if _isOpaqueZero() {
      return type(of: x)._zero._subtracting(x)
    }
    guard let xBase = x._unboxed(to: T.self) else {
      _derivativeTypeMismatch(T.self, type(of: x._typeErasedBase))
    }
    return _ConcreteDerivativeBox(_base - xBase)
  }

  // `Differentiable` requirements.
  @inlinable
  mutating func _move(along direction: _AnyDerivativeBox) {
    if direction._isOpaqueZero() {
      return
    }
    // The case where `self._isOpaqueZero()` returns true is handled in
    // `AnyDerivative.move(along:)`.
    guard
      let directionBase =
        direction._unboxed(to: T.TangentVector.self)
    else {
      _derivativeTypeMismatch(T.self, type(of: direction._typeErasedBase))
    }
    _base.move(along: directionBase)
  }
}

/// A type-erased derivative value.
///
/// The `AnyDerivative` type forwards its operations to an arbitrary underlying
/// base derivative value conforming to `Differentiable` and
/// `AdditiveArithmetic`, hiding the specifics of the underlying value.
@frozen
public struct AnyDerivative: Differentiable & AdditiveArithmetic {
  @usableFromInline
  internal var _box: _AnyDerivativeBox

  @inlinable
  internal init(_box: _AnyDerivativeBox) {
    self._box = _box
  }

  /// The underlying base value.
  @inlinable
  public var base: Any {
    return _box._typeErasedBase
  }

  /// Creates a type-erased derivative from the given derivative.
  @inlinable
  @differentiable
  public init<T>(_ base: T) where T: Differentiable, T.TangentVector == T {
    self._box = _ConcreteDerivativeBox<T>(base)
  }

  @inlinable
  @derivative(of: init)
  internal static func _vjpInit<T>(
    _ base: T
  ) -> (value: AnyDerivative, pullback: (AnyDerivative) -> T.TangentVector)
  where T: Differentiable, T.TangentVector == T {
    return (AnyDerivative(base), { v in v.base as! T.TangentVector })
  }

  @inlinable
  @derivative(of: init)
  internal static func _jvpInit<T>(
    _ base: T
  ) -> (value: AnyDerivative, differential: (T.TangentVector) -> AnyDerivative)
  where T: Differentiable, T.TangentVector == T {
    return (AnyDerivative(base), { dbase in AnyDerivative(dbase) })
  }

  public typealias TangentVector = AnyDerivative

  // `Equatable` requirements (implied by `AdditiveArithmetic`).
  @inlinable
  public static func == (lhs: AnyDerivative, rhs: AnyDerivative) -> Bool {
    return lhs._box._isEqual(to: rhs._box)
  }
  @inlinable
  public static func != (lhs: AnyDerivative, rhs: AnyDerivative) -> Bool {
    return lhs._box._isNotEqual(to: rhs._box)
  }

  // `AdditiveArithmetic` requirements.

  /// Internal struct representing an opaque zero value.
  @frozen
  @usableFromInline
  internal struct OpaqueZero: Differentiable & AdditiveArithmetic {}

  @inlinable
  public static var zero: AnyDerivative {
    return AnyDerivative(
      _box: _ConcreteDerivativeBox<OpaqueZero>(OpaqueZero.zero))
  }

  @inlinable
  public static func + (
    lhs: AnyDerivative, rhs: AnyDerivative
  ) -> AnyDerivative {
    return AnyDerivative(_box: lhs._box._adding(rhs._box))
  }

  @derivative(of: +)
  @inlinable
  internal static func _vjpAdd(
    lhs: AnyDerivative, rhs: AnyDerivative
  ) -> (
    value: AnyDerivative,
    pullback: (AnyDerivative) -> (AnyDerivative, AnyDerivative)
  ) {
    return (lhs + rhs, { v in (v, v) })
  }

  @derivative(of: +)
  @inlinable
  internal static func _jvpAdd(
    lhs: AnyDerivative, rhs: AnyDerivative
  ) -> (
    value: AnyDerivative,
    differential: (AnyDerivative, AnyDerivative) -> (AnyDerivative)
  ) {
    return (lhs + rhs, { (dlhs, drhs) in dlhs + drhs })
  }

  @inlinable
  public static func - (
    lhs: AnyDerivative, rhs: AnyDerivative
  ) -> AnyDerivative {
    return AnyDerivative(_box: lhs._box._subtracting(rhs._box))
  }

  @derivative(of: -)
  @inlinable
  internal static func _vjpSubtract(
    lhs: AnyDerivative, rhs: AnyDerivative
  ) -> (
    value: AnyDerivative,
    pullback: (AnyDerivative) -> (AnyDerivative, AnyDerivative)
  ) {
    return (lhs - rhs, { v in (v, .zero - v) })
  }

  @derivative(of: -)
  @inlinable
  internal static func _jvpSubtract(
    lhs: AnyDerivative, rhs: AnyDerivative
  ) -> (
    value: AnyDerivative,
    differential: (AnyDerivative, AnyDerivative) -> AnyDerivative
  ) {
    return (lhs - rhs, { (dlhs, drhs) in dlhs - drhs })
  }

  // `Differentiable` requirements.
  @inlinable
  public mutating func move(along direction: TangentVector) {
    if _box._isOpaqueZero() {
      _box = direction._box
      return
    }
    _box._move(along: direction._box)
  }
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

@inline(never)
@usableFromInline
internal func _derivativeTypeMismatch(
  _ x: Any.Type, _ y: Any.Type, file: StaticString = #file, line: UInt = #line
) -> Never {
  preconditionFailure(
    """
    Derivative type mismatch: \
    \(String(reflecting: x)) and \(String(reflecting: y))
    """, file: file, line: line)
}
