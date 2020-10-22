extension Vector5: ManifoldCoordinate {
  /// The local coordinate type of the manifold.
  ///
  /// This is the `TangentVector` of the `Manifold` wrapper type.
  ///
  /// Note that this is not the same type as `Self.TangentVector`.
  public typealias LocalCoordinate = Self

  /// Diffeomorphism between a neigborhood of `LocalCoordinate.zero` and `Self`.
  ///
  /// Satisfies the following properties:
  /// - `retract(LocalCoordinate.zero) == self`
  /// - There exists an open set `B` around `LocalCoordinate.zero` such that
  ///   `localCoordinate(retract(b)) == b` for all `b \in B`.
  @differentiable(wrt: local)
  public func retract(_ local: LocalCoordinate) -> Self {
    self + local
  }

  /// Inverse of `retract`.
  ///
  /// Satisfies the following properties:
  /// - `localCoordinate(self) == LocalCoordinate.zero`
  /// - There exists an open set `B` around `self` such that `localCoordinate(retract(b)) == b` for all
  ///   `b \in B`.
  @differentiable(wrt: global)
  public func localCoordinate(_ global: Self) -> LocalCoordinate {
    global - self
  }
}

extension Vector5: Manifold {
  /// The manifold's global coordinate system.
  public typealias Coordinate = Self

  /// The coordinate of `self`.
  ///
  /// Note: The distinction between `coordinateStorage` and `coordinate` is a workaround until we
  /// can define default derivatives for protocol requirements (TF-982). Until then, implementers
  /// of this protocol must define `coordinateStorage`, and clients of this protocol must access
  /// coordinate`. This allows us to define default derivatives for `coordinate` that translate
  /// between the `ManifoldCoordinate` tangent space and the `Manifold` tangent space.
  public var coordinateStorage: Coordinate {
    get {
      self
    }

    set {
      self = newValue
    }
  }

  /// Creates a manifold point with coordinate `coordinateStorage`.
  ///
  /// Note: The distinction between `init(coordinateStorage:)` and `init(coordinate:)` is a workaround until we
  /// can define default derivatives for protocol requirements (TF-982). Until then, implementers
  /// of this protocol must define `init(coordinateStorage:)`, and clients of this protocol must access
  /// init(coordinate:)`. This allows us to define default derivatives for `init(coordinate:)` that translate
  /// between the `ManifoldCoordinate` tangent space and the `Manifold` tangent space.
  public init(coordinateStorage: Coordinate) {
    self = coordinateStorage
  }
}

extension Vector5: LieGroupCoordinate {
  /// Creates the group identity.
  public init() {
    self = Self.zero
  }

  /// Returns the group inverse.
  @differentiable
  public func inverse() -> Self {
    -self
  }

  /// The group operation.
  @differentiable
  public static func * (_ lhs: Self, _ rhs: Self) -> Self {
    lhs + rhs
  }

  public func AdjointTranspose(_ v: LocalCoordinate) -> LocalCoordinate {
    return defaultAdjointTranspose(v)
  }
}

extension Vector5: LieGroup {}


extension Vector7: ManifoldCoordinate {
  public typealias LocalCoordinate = Self

  @differentiable(wrt: local)
  public func retract(_ local: LocalCoordinate) -> Self {
    self + local
  }

  @differentiable(wrt: global)
  public func localCoordinate(_ global: Self) -> LocalCoordinate {
    global - self
  }
}

extension Vector7: Manifold {
  /// The manifold's global coordinate system.
  public typealias Coordinate = Self

  public var coordinateStorage: Coordinate {
    get {
      self
    }

    set {
      self = newValue
    }
  }

  /// Creates a manifold point with coordinate `coordinateStorage`.
  public init(coordinateStorage: Coordinate) {
    self = coordinateStorage
  }
}

extension Vector7: LieGroupCoordinate {
  /// Creates the group identity.
  public init() {
    self = Self.zero
  }

  /// Returns the group inverse.
  @differentiable
  public func inverse() -> Self {
    -self
  }

  /// The group operation.
  @differentiable
  public static func * (_ lhs: Self, _ rhs: Self) -> Self {
    lhs + rhs
  }

  public func AdjointTranspose(_ v: LocalCoordinate) -> LocalCoordinate {
    return defaultAdjointTranspose(v)
  }
}

extension Vector7: LieGroup {}

// -------------------------------------------------------------

/// MARK: Vector10
extension Vector10: ManifoldCoordinate {
  public typealias LocalCoordinate = Self

  @differentiable(wrt: local)
  public func retract(_ local: LocalCoordinate) -> Self {
    self + local
  }

  @differentiable(wrt: global)
  public func localCoordinate(_ global: Self) -> LocalCoordinate {
    global - self
  }
}

extension Vector10: Manifold {
  /// The manifold's global coordinate system.
  public typealias Coordinate = Self

  public var coordinateStorage: Coordinate {
    get {
      self
    }

    set {
      self = newValue
    }
  }

  /// Creates a manifold point with coordinate `coordinateStorage`.
  public init(coordinateStorage: Coordinate) {
    self = coordinateStorage
  }
}

extension Vector10: LieGroupCoordinate {
  /// Creates the group identity.
  public init() {
    self = Self.zero
  }

  /// Returns the group inverse.
  @differentiable
  public func inverse() -> Self {
    -self
  }

  /// The group operation.
  @differentiable
  public static func * (_ lhs: Self, _ rhs: Self) -> Self {
    lhs + rhs
  }

  public func AdjointTranspose(_ v: LocalCoordinate) -> LocalCoordinate {
    return defaultAdjointTranspose(v)
  }
}

extension Vector10: LieGroup {}
