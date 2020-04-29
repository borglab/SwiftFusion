/// A coordinate in a manifold's global coordinate system.
public protocol ManifoldGlobalCoordinate: Differentiable {
  /// The local coordinate system used in the chart.
  ///
  /// Isomorphic to `R^n`, where `n` is the dimension of the manifold.
  associatedtype LocalCoordinate: AdditiveArithmetic & Differentiable & VectorProtocol where LocalCoordinate.TangentVector == LocalCoordinate

  /// The global coordinate corresponding to `local` in the chart centered around `self`.
  ///
  /// Satisfies the following properties:
  /// - `global(LocalCoordinate.zero) == coordinate`
  /// - There exists an open ball `B` around `LocalCoordinate.zero` such that `local(global(b)) == b` for all
  ///   `b \in B`.
  @differentiable(wrt: local)
  func global(_ local: LocalCoordinate) -> Self

  /// The local coordinate corresponding to `global` in the chart centered around `self`.
  ///
  /// Satisfies the following properties:
  /// - `local(coordinate) == LocalCoordinate.zero`
  /// - There exists an open ball `B` around `self` such that `local(global(b)) == b` for all
  ///   `b \in B`.
  @differentiable(wrt: global)
  func local(_ global: Self) -> LocalCoordinate
}

public protocol Manifold: Differentiable {
  associatedtype GlobalCoordinate: ManifoldGlobalCoordinate where GlobalCoordinate.LocalCoordinate == Self.TangentVector
  var whatToNameTheCoordinate: GlobalCoordinate { get set }
  init(whatToNameTheCoordinate: GlobalCoordinate)
}

extension Manifold {
  @differentiable
  public var coordinate: GlobalCoordinate { whatToNameTheCoordinate }

  @derivative(of: coordinate)
  @usableFromInline
  func vjpCoordinate() -> (value: GlobalCoordinate, pullback: (GlobalCoordinate.TangentVector) -> TangentVector) {
    return (
      value: coordinate,
      pullback(at: GlobalCoordinate.LocalCoordinate.zero) { self.whatToNameTheCoordinate.global($0) }
    )
  }

  @differentiable
  public init(coordinate: GlobalCoordinate) { self.init(whatToNameTheCoordinate: coordinate) }

  @derivative(of: init(coordinate:))
  @usableFromInline
  static func vjpInit(coordinate: GlobalCoordinate) -> (value: Self, pullback: (TangentVector) -> GlobalCoordinate.TangentVector) {
    return (
      value: Self(coordinate: coordinate),
      pullback: pullback(at: coordinate) { coordinate.local($0) }
    )
  }
}
