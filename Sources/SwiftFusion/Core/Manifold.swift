/// A coordinate in a differentiable manifold's global coordinate system.
///
/// Includes a coordinate chart [1] between `Self` and `R^n`, centered at `self`.
///
/// [1] https://en.wikipedia.org/wiki/Topological_manifold#Coordinate_charts
public protocol ManifoldCoordinate: Differentiable {
  /// The local coordinate system used in the chart.
  ///
  /// Isomorphic to `R^n`, where `n` is the dimension of the manifold.
  associatedtype LocalCoordinate: AdditiveArithmetic & Differentiable & VectorProtocol & TensorConvertible & TangentStandardBasis
    where LocalCoordinate.TangentVector == LocalCoordinate

  /// The global coordinate corresponding to `local` in the chart centered around `self`.
  ///
  /// Satisfies the following properties:
  /// - `global(LocalCoordinate.zero) == self`
  /// - There exists an open set `B` around `LocalCoordinate.zero` such that
  ///   `local(global(b)) == b` for all `b \in B`.
  @differentiable(wrt: local)
  func global(_ local: LocalCoordinate) -> Self

  /// The local coordinate corresponding to `global` in the chart centered around `self`.
  ///
  /// Satisfies the following properties:
  /// - `local(self) == LocalCoordinate.zero`
  /// - There exists an open set `B` around `self` such that `local(global(b)) == b` for all
  ///   `b \in B`.
  @differentiable(wrt: global)
  func local(_ global: Self) -> LocalCoordinate
}

/// A point on a differentiable manifold.
public protocol Manifold: Differentiable {
  /// The manifold's global coordinate system.
  associatedtype Coordinate: ManifoldCoordinate
      where Coordinate.LocalCoordinate == Self.TangentVector

  /// The coordinate of `self`.
  ///
  /// Note: This is not differentiable, and therefore clients should use `coordinate` instead.
  var coordinateStorage: Coordinate { get set }

  /// Creates a manifold point with coordinate `coordinateStorage`.
  ///
  /// Note: This is not differentiable, and therefore clients should use `init(coordinate:)`
  /// instead.
  init(coordinateStorage: Coordinate)
}

extension Manifold {
  /// The coordinate of `self`.
  @differentiable
  public var coordinate: Coordinate { coordinateStorage }

  /// A custom derivative of `coordinate` that converts from the global coordinate system's
  /// tangent vector to the local coordinate system's tangent vector, so that all functions on this
  /// manifold using `coordinate` have derivatives involving local coordinates.
  @derivative(of: coordinate)
  @usableFromInline
  func vjpCoordinate() -> (value: Coordinate, pullback: (Coordinate.TangentVector) -> TangentVector) {

    // Explanation of this pullback:
    //
    // Let `f: Manifold -> Coordinate` be `f(x) = x.coordinateStorage`.
    //
    // `D_x(f)` (the derivative of `f` at `x`) is a linear approximation of how changes in local
    // coordinates around `x` lead to changes in global coordinates around `x.coordinateStorage`.
    //
    // `x.coordinateStorage.global: LocalCoordinate -> Coordinate` defines _exactly_ how local
    // coordinates at `x` map to global coordinates.
    //
    // Therefore, `D_x(f)` is the derivative of `x.coordinateStorage.global`.

    return (
      value: coordinateStorage,
      pullback(at: Coordinate.LocalCoordinate.zero) { self.coordinateStorage.global($0) }
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
  static func vjpInit(coordinate: Coordinate) -> (value: Self, pullback: (TangentVector) -> Coordinate.TangentVector) {

    // Explanation of this pullback:
    //
    // Let `g: Coordinate -> Manifold` be `g(x) = Self(coordinateStorage: x)`.
    //
    // `D_x(g)` (the derivative of `g` at `x`) is a linear approximation of how changes in global
    // coordinates around `x` lead to changes in local coordinates around
    // `Self(coordinateStorage: x)`.
    //
    // `x.coordinateStorage.local: Coordinate -> LocalCoordinate` defines _exactly_ how global
    // coordinates around `x` map to local coordinates.
    //
    // Therefore, `D_x(g)` is the derivative of `x.coordinateStorage.local`.

    return (
      value: Self(coordinateStorage: coordinate),
      pullback: pullback(at: coordinate) { coordinate.local($0) }
    )
  }
}
