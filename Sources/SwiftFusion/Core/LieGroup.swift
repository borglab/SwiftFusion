import _Differentiation
import TensorFlow

/// An element of a Lie group.
///
/// To create a Lie group type, follow the manifold recipe [1] and conform the coordinate type to
/// `LieGroupCoordinate`. Then, conform your manifold type to `LieGroup` and it will automatically
/// get Lie group operations.
///
/// [1] SwiftFusion/doc/DifferentiableManifoldRecipe.md
public protocol LieGroup: Manifold
  where Coordinate: LieGroupCoordinate, TangentVector == Coordinate.LocalCoordinate {}

/// Default implementations of group operations in terms of the corresponding coordinate operations.
extension LieGroup {
  /// Creates the group identity.
  public init() {
    self.init(coordinate: Coordinate())
  }

  /// Returns the group inverse.
  @differentiable(wrt: self)
  public func inverse() -> Self {
    return Self(coordinate: self.coordinate.inverse())
  }

  /// Derivative of `inverse`.
  ///
  /// Swift AD can compute this, but we know a mathematical expression for the derivative that
  /// is more efficient than the compiler can figure out.
  @derivative(of: inverse)
  @usableFromInline
  func vjpInverse() -> (value: Self, pullback: (TangentVector) -> TangentVector) {
    // Derivative from https://github.com/borglab/gtsam/blob/develop/doc/math.pdf section 5.3. We
    // use the transpose of the Adjoint because the pullback is the transpose of the differential.
    return (self.inverse(), { -1 * self.AdjointTranspose($0) })
  }

  /// The group operation.
  @differentiable(wrt: (lhs, rhs))
  @differentiable(wrt: lhs)
  @differentiable(wrt: rhs)
  public static func * (_ lhs: Self, _ rhs: Self) -> Self {
    return Self(coordinate: lhs.coordinate * rhs.coordinate)
  }

  /// Derivative of `*` with respect to both sides.
  ///
  /// We define separate derivatives with respect to different sets of arguments so that we can
  /// avoid doing unnecessary work when some of the arguments are not changing.
  ///
  /// Swift AD can compute this, but we know a mathematical expression for the derivative that
  /// is more efficient than the compiler can figure out.
  @derivative(of: *, wrt: (lhs, rhs))
  @usableFromInline
  static func vjpGroupOperationWrtBoth(_ lhs: Self, _ rhs: Self) ->
    (value: Self, pullback: (TangentVector) -> (TangentVector, TangentVector))
  {
    // Derivative from https://github.com/borglab/gtsam/blob/develop/doc/math.pdf sections 5.2 and
    // 5.4. We use the transpose of the Adjoint because the pullback is the transpose of the
    // differential.
    return (lhs * rhs, { (rhs.inverse().AdjointTranspose($0), $0) })
  }

  /// Derivative of `*` with respect to the left side.
  ///
  /// We define separate derivatives with respect to different sets of arguments so that we can
  /// avoid doing unnecessary work when some of the arguments are not changing.
  ///
  /// Swift AD can compute this, but we know a mathematical expression for the derivative that
  /// is more efficient than the compiler can figure out.
  @derivative(of: *, wrt: lhs)
  @usableFromInline
  static func vjpGroupOperationWrtLhs(_ lhs: Self, _ rhs: Self) ->
    (value: Self, pullback: (TangentVector) -> TangentVector)
  {
    // Derivative from https://github.com/borglab/gtsam/blob/develop/doc/math.pdf section 5.4. We
    // use the transpose of the Adjoint because the pullback is the transpose of the differential.
    return (lhs * rhs, { rhs.inverse().AdjointTranspose($0) })
  }

  /// Derivative of `*` with respect to the right side.
  ///
  /// We define separate derivatives with respect to different sets of arguments so that we can
  /// avoid doing unnecessary work when some of the arguments are not changing.
  ///
  /// Swift AD can compute this, but we know a mathematical expression for the derivative that
  /// is more efficient than the compiler can figure out.
  @derivative(of: *, wrt: rhs)
  @usableFromInline
  static func vjpGroupOperationWrtRhs(_ lhs: Self, _ rhs: Self) ->
    (value: Self, pullback: (TangentVector) -> TangentVector)
  {
    // Derivative from https://github.com/borglab/gtsam/blob/develop/doc/math.pdf section 5.2. We
    // use the transpose of the Adjoint because the pullback is the transpose of the differential.
    return (lhs * rhs, { $0 })
  }

  /// The Adjoint group action of `self` on `v`.
  public func Adjoint(_ v: TangentVector) -> TangentVector {
    return coordinate.Adjoint(v)
  }

  /// The transpose of the Adjoint group action of `self` on `v`.
  public func AdjointTranspose(_ v: TangentVector) -> TangentVector {
    return coordinate.AdjointTranspose(v)
  }
}

/// The `ManifoldCoordinate` of a `LieGroup`.
public protocol LieGroupCoordinate: ManifoldCoordinate {
  /// Creates the group identity.
  init()

  /// Returns the group inverse.
  @differentiable(wrt: self)
  func inverse() -> Self

  /// The group operation.
  @differentiable(wrt: (lhs, rhs))
  static func * (_ lhs: Self, _ rhs: Self) -> Self

  /// The Adjoint group action of `self` on `v`.
  ///
  /// A default implementation in terms of other group and manifold operations is provided.
  /// Implementers may wish to provide a more efficient implementation, because the compiler may
  /// not have enough knowledge about the mathematical structure to simplify the default
  /// implementation as much as possible.
  ///
  /// This is differentiable with respect to `v` so that we can use a `pullback` to transpose it in
  /// the default implementation for `AdjointTranspose`.
  @differentiable(wrt: v)
  func Adjoint(_ v: LocalCoordinate) -> LocalCoordinate

  /// The transpose of the Adjoint group action of `self` on `v`.
  ///
  /// A default implementation in terms of other group and manifold operations is provided.
  /// Implementers may wish to provide a more efficient implementation, because the compiler may
  /// not have enough knowledge about the mathematical structure to simplify the default
  /// implementation as much as possible.
  func AdjointTranspose(_ v: LocalCoordinate) -> LocalCoordinate
}

/// Default implementations of `Adjoint` and `AdjointTranspose` in terms of other group
/// operations.
extension LieGroupCoordinate {
  @differentiable(wrt: v)
  public func Adjoint(_ v: LocalCoordinate) -> LocalCoordinate {
    return defaultAdjoint(v)
  }

  public func AdjointTranspose(_ v: LocalCoordinate) -> LocalCoordinate {
    return defaultAdjointTranspose(v)
  }

  /// The default implementation of `Adjoint`, provided so that implementers can test their
  /// implementation against the default implementation.
  @differentiable(wrt: v)
  public func defaultAdjoint(_ v: LocalCoordinate) -> LocalCoordinate {
    let identity = Self()
    func log(_ g: Self) -> LocalCoordinate { return identity.localCoordinate(g) }
    func exp(_ v: LocalCoordinate) -> Self { return identity.retract(v) }
    return log(self * exp(v) * self.inverse())
  }

  /// The default implementation of `AdjointTranspose`, provided so that implementers can test
  /// their implementation against the default implementation.
  public func defaultAdjointTranspose(_ v: LocalCoordinate) -> LocalCoordinate {
    // This works because the pullback of a linear function is its transpose.
    return pullback(at: v.zeroTangentVector, in: { self.Adjoint($0) })(v)
  }
}

/// Calculate relative pose 1T2 between two poses wT1 and wT2
@differentiable(wrt: (wT1, wT2))
public func between<T: LieGroup & Differentiable>(_ wT1: T, _ wT2: T) -> T {
  wT1.inverse() * wT2
}
