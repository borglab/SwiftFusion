/// A point on a differentiable manifold with a `retract` map centered around `self`.
///
/// This protocol helps you define manifolds with custom tangent vectors. Instructions:
/// 1. Define a type `C: ManifoldCoordinate`, without specifying a `TangentVector`. (Swift
///    generates a `TangentVector` automatically, usually not the `TangentVector` that you want for
///    your manifold).
/// 2. Define `C.LocalCoordinate` to be the `TangentVector` type that you want for your manifold,
///    and define `C.retract` and `C.localCoordinate` to be the retraction and inverse retration
///    for this `TangentVector`.
/// 3. Define a type `M: Manifold` that wraps `C`. The `Manifold` protocol automatically gives `M`
///    the desired `TangentVector`.
/// See "SwiftFusion/doc/DifferentiableManifoldRecipe.md" for more detailed instructions.
public protocol ManifoldCoordinate: Differentiable {
  /// The local coordinate type of the manifold.
  ///
  /// This is the `TangentVector` of the `Manifold` wrapper type.
  ///
  /// Note that this is not the same type as `Self.TangentVector`.
  associatedtype LocalCoordinate: EuclideanVector & TangentStandardBasis

  /// Diffeomorphism between a neigborhood of `LocalCoordinate.zero` and `Self`.
  ///
  /// Satisfies the following properties:
  /// - `retract(LocalCoordinate.zero) == self`
  /// - There exists an open set `B` around `LocalCoordinate.zero` such that
  ///   `localCoordinate(retract(b)) == b` for all `b \in B`.
  @differentiable(wrt: local)
  func retract(_ local: LocalCoordinate) -> Self

  /// Inverse of `retract`.
  ///
  /// Satisfies the following properties:
  /// - `localCoordinate(self) == LocalCoordinate.zero`
  /// - There exists an open set `B` around `self` such that `localCoordinate(retract(b)) == b` for all
  ///   `b \in B`.
  @differentiable(wrt: global)
  func localCoordinate(_ global: Self) -> LocalCoordinate
}

/// A point on a differentiable manifold.
public protocol Manifold: Differentiable {
  /// The manifold's global coordinate system.
  associatedtype Coordinate: ManifoldCoordinate

  /// The coordinate of `self`.
  ///
  /// Note: The distinction between `coordinateStorage` and `coordinate` is a workaround until we
  /// can define default derivatives for protocol requirements (TF-982). Until then, implementers
  /// of this protocol must define `coordinateStorage`, and clients of this protocol must access
  /// coordinate`. This allows us to define default derivatives for `coordinate` that translate
  /// between the `ManifoldCoordinate` tangent space and the `Manifold` tangent space.
  var coordinateStorage: Coordinate { get set }

  /// Creates a manifold point with coordinate `coordinateStorage`.
  ///
  /// Note: The distinction between `init(coordinateStorage:)` and `init(coordinate:)` is a workaround until we
  /// can define default derivatives for protocol requirements (TF-982). Until then, implementers
  /// of this protocol must define `init(coordinateStorage:)`, and clients of this protocol must access
  /// init(coordinate:)`. This allows us to define default derivatives for `init(coordinate:)` that translate
  /// between the `ManifoldCoordinate` tangent space and the `Manifold` tangent space.
  init(coordinateStorage: Coordinate)
}

/// Methods for converting between manifolds and their coordinates.
///
/// To enable these, you must explicitly write
//    public typealias TangentVector = <local coordinate type>
/// in your manifold type.
extension Manifold where Self.TangentVector == Coordinate.LocalCoordinate {
  /// The coordinate of `self`.
  @differentiable
  public var coordinate: Coordinate {
    return coordinateStorage
  }

  /// A custom derivative of `coordinate` that converts from the global coordinate system's
  /// tangent vector to the local coordinate system's tangent vector, so that all functions on this
  /// manifold using `coordinate` have derivatives involving local coordinates.
  @derivative(of: coordinate)
  @usableFromInline
  func vjpCoordinate()
    -> (value: Coordinate, pullback: (Coordinate.TangentVector) -> TangentVector)
  {

    // Explanation of this pullback:
    //
    // Let `f: Manifold -> Coordinate` be `f(x) = x.coordinateStorage`.
    //
    // `differential(at: x, in: f)` is a linear approximation of how changes in tangent vectors
    // around `x` lead to changes in global coordinates around `x.coordinateStorage`.
    //
    // `x.coordinateStorage.retract: TangentVector -> Coordinate` defines _exactly_ how local
    // coordinates around zero map to global coordinates around `x`.
    //
    // Therefore, `differential(at: x, in: f) = differential(at: zero, in: x.coordinateStorage.retract)`.
    //
    // The pullback is the dual map of the differential, so taking duals of both sides gives:
    //   `pullback(at: x, in: f) = pullback(at: zero, in: x.coordinateStorage.retract)`.

    return (
      value: coordinateStorage,
      pullback(at: Coordinate.LocalCoordinate.zero) { self.coordinateStorage.retract($0) }
    )
  }

  /// Creates a manifold point with coordinate `coordinate`.
  @differentiable
  public init(coordinate: Coordinate) { self.init(coordinateStorage: coordinate) }

  /// A custom derivative of `init(coordinate:)` that converts from the local coordinate system's
  /// tangent vector to the global coordinate system's tangent vector, so that all functions
  /// producing instances of this manifold using `init(coordinates:)` have derivatives involving
  /// local coordinates.
  @derivative(of: init(coordinate:))
  @usableFromInline
  static func vjpInit(coordinate: Coordinate)
    -> (value: Self, pullback: (TangentVector) -> Coordinate.TangentVector)
  {

    // Explanation of this pullback:
    //
    // Let `g: Coordinate -> Manifold` be `g(x) = Self(coordinateStorage: x)`.
    //
    // `D_x(g)` (the derivative of `g` at `x`) is a linear approximation of how changes in global
    // coordinates around `x` lead to changes in tangent vectors around
    // `Self(coordinateStorage: x)`.
    //
    // `x.coordinateStorage.localCoordinate: Coordinate -> TangentVector` defines _exactly_ how global
    // coordinates around `x` map to tangent vectors.
    //
    // Therefore, `D_x(g)` is the derivative of `x.coordinateStorage.localCoordinate`.

    // Explanation of this pullback:
    //
    // Let `g: Coordinate -> Manifold` be `g(x) = Self(coordinateStorage: x)`.
    //
    // `differential(at: x, in: g)` is a linear approximation of how changes in global
    // coordinates around `x` lead to changes in local coordinates around `Self(coordinateStorage: x)`.
    //
    // `x.coordinateStorage.localCoordinate: Coordinate -> LocalCoordinate` defines _exactly_ how global
    // coordinates around `x` map to local coordinates.
    //
    // Therefore, `differential(at: x, in: g) = differential(at: zero, in: x.coordinateStorage.localCoordinate)`.
    //
    // The pullback is the dual map of the differential, so taking duals of both sides gives:
    //   `pullback(at: x, in: g) = pullback(at: zero, in: x.coordinateStorage.localCoordinate)`.

    return (
      value: Self(coordinateStorage: coordinate),
      pullback: pullback(at: coordinate) { coordinate.localCoordinate($0) }
    )
  }
}

/// Default implementations of manifold operations in terms of the corresponding
/// `ManifoldCoordinate` operations.
extension Manifold where Self.TangentVector == Coordinate.LocalCoordinate {
  /// Diffeomorphism between a neigborhood of `TangentVector.zero` and `Self`.
  ///
  /// Satisfies the following properties:
  /// - `retract(TangentVector.zero) == self`
  /// - There exists an open set `B` around `TangentVector.zero` such that
  ///   `localCoordinate(retract(b)) == b` for all `b \in B`.
  @differentiable(wrt: local)
  public func retract(_ local: TangentVector) -> Self {
    return Self(coordinate: self.coordinate.retract(local))
  }

  /// Derivative of `retract`.
  ///
  /// Swift AD can compute this, but we know mathematically that the derivative is the identity, so
  /// we can provide an implementation that is more efficient.
  @derivative(of: retract, wrt: local)
  @usableFromInline
  func vjpRetract(_ local: TangentVector) -> (value: Self, pullback: (TangentVector) -> TangentVector) {
    return (retract(local), { $0 })
  }

  /// Inverse of `retract`.
  ///
  /// Satisfies the following properties:
  /// - `localCoordinate(self) == TangentVector.zero`
  /// - There exists an open set `B` around `self` such that `localCoordinate(retract(b)) == b` for all
  ///   `b \in B`.
  @differentiable(wrt: global)
  public func localCoordinate(_ global: Self) -> TangentVector {
    return self.coordinate.localCoordinate(global.coordinate)
  }

  /// Derivative of `localCoordinate`.
  ///
  /// Swift AD can compute this, but we know mathematically that the derivative is the identity, so
  /// we can provide an implementation that is more efficient.
  @derivative(of: localCoordinate, wrt: global)
  @usableFromInline
  func vjpLocalCoordinate(_ global: Self) ->
    (value: TangentVector, pullback: (TangentVector) -> TangentVector)
  {
    return (localCoordinate(global), { $0 })
  }
}
