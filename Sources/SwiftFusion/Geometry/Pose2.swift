import TensorFlow

/// SE(2) Lie group of 2D Euclidean Poses.
///
/// We adopt the following conventions for the tangent space of this manifold and the derivatives
/// of functions to/from this manifold:
///
///   1. The tangent space at every point is R^3, interpreted as "twist coordinates", with the
///      first component specifying the rotation and the next two coordinates specifying the
///      translation.
///   2. For Lie groups `G`, `H`, with algebras `g`, `h`, the differential of any differentiable
///      function `f : G -> H` at `a \in G` is the linear map `df_a : g -> h` such that
///        f(a * exp(hat(eps))) ~= f(a) * exp(df_a(hat(eps)))
///      where:
///        `eps` is a "small" element of `R^3`, and
///        `hat` is the map from twist coordinates to se(2).
///      The pullback of `f` at `a` is the dual map of `df_a`.
///
/// These conventions match the conventions for SE(2) in [1] and [2], except that we use the
/// opposite order for the rotation and translation components.
///
/// (1) is accomplished by setting the `TangentVector` to `Vector3` and by defining the appropriate
/// exponential map in the `move(along:)` method.
///
/// (2) is accomplished for `Pose2` by defining differentials[3] satisfying (2) for three
//  functions:
///   - .t: SE(2) -> R^2
///   - .rot: SE(2) -> SO(2)
///   - init: R^2 x SO(2) -> SE(2)
/// Since these are the only functions exposed by the original `Pose2` declaration, all other
/// functions involving `Pose2` are forced to be compositions of these. Therefore, the derivatives
/// synthesized by Swift's automatic differentiation transform satisfy (2).
///
/// [1]: https://github.com/borglab/gtsam/blob/develop/doc/LieGroups.pdf
/// [2]: https://github.com/borglab/gtsam/blob/develop/doc/math.pdf
/// [3]: Actually, we define the pullbacks because Swift doesn't support differentials very well
///      yet.
public struct Pose2: Manifold, Equatable, TangentStandardBasis, KeyPathIterable {
  // MARK: - Manifold conformance

  public var coordinateStorage: Pose2Coordinate
  public init(coordinateStorage: Pose2Coordinate) { self.coordinateStorage = coordinateStorage }

  public mutating func move(along direction: Coordinate.LocalCoordinate) {
    coordinateStorage = coordinateStorage.global(direction)
  }

  /// Creates a `Pose2` with rotation `r` and translation `t`.
  ///
  /// This is the bijection SO(2) x R^2 -> SE(2), where "x" means direct product of groups. (Note:
  /// not a group homomorphism!)
  @differentiable
  public init(_ r: Rot2, _ t: Vector2) {
    self.init(coordinate: Pose2Coordinate(r, t))
  }
  
  // MARK: Convenience Attributes
  
  @differentiable public var t: Vector2 { coordinate.t }
  
  @differentiable public var rot: Rot2 { coordinate.rot }
  
  /// Twist coordinates, with the first component for rotation and the remaining components for
  /// translation.
  ///
  /// See the documentation comment on the declaration of `Pose2` for more information.
  public typealias TangentVector = Vector3

  /// Product of two rotations.
  @differentiable
  public static func * (lhs: Pose2, rhs: Pose2) -> Pose2 {
    Pose2(coordinate: lhs.coordinate * rhs.coordinate)
  }
  
  @differentiable
  public func inverse() -> Pose2 {
    Pose2(coordinate: coordinate.inverse())
  }
}

// MARK: Convenience initializers
extension Pose2 {
  /// Creates a `Pose2` with translation `x` and `y` and with rotation `theta`.
  @differentiable
  public init(_ x: Double, _ y: Double, _ theta: Double) {
    self.init(Rot2(theta), Vector2(x, y))
  }

  public init(randomWithCovariance covariance: Tensor<Double>) {
    self.init(0, 0, 0)
    let r = matmul(cholesky(covariance), Tensor<Double>(randomNormal: [3, 1])).scalars
    let tv = Pose2.TangentVector(r[0], r[1], r[2])
    self.move(along: tv)
  }
}

// MARK: - Global Coordinate System

public struct Pose2Coordinate: Equatable, KeyPathIterable {
  var tStorage: Vector2
  var rotStorage: Rot2
  
  @differentiable
  public var t: Vector2 { tStorage }
  
  @differentiable
  public var rot: Rot2 { rotStorage }
  
  /// Twist coordinates, with the first component for rotation and the remaining components for
  /// translation.
  ///
  /// See the documentation comment on the declaration of `Pose2` for more information.
  public typealias TangentVector = Vector3
}

public extension Pose2Coordinate {
  @differentiable
  init(_ rot: Rot2, _ t: Vector2) {
    self.tStorage = t
    self.rotStorage = rot
  }

  /// Returns the rotation (`w`) and translation (`v`) components of `tangentVector`.
  static func decomposed(tangentVector: Vector3) -> (w: Vector1, v: Vector2) {
    (Vector1(tangentVector.x), Vector2(tangentVector.y, tangentVector.z))
  }
  
  /// Creates a tangent vector given rotation (`w`) and translation (`v`) components.
  static func tangentVector(w: Vector1, v: Vector2) -> Vector3 {
    Vector3(w.x, v.x, v.y)
  }
}

// MARK: Custom Derivatives

extension Pose2Coordinate {
  /// The derivative of `init` satisfiying convention (2) described above.
  @derivative(of: Pose2Coordinate.init(_:_:))
  @usableFromInline
  static func vjpInit(_ r: Rot2, _ t: Vector2)
    -> (value: Pose2Coordinate, pullback: (TangentVector) -> (Rot2.TangentVector, Vector2))
  {
    // Explanation of this calculation:
    //
    // We would like the differential of `init` at `(r, t)`, `dinit_(r, t)`, to satisfy:
    //   init(r * exp(hat(w)), t * exp(hat(v))) ~= init(r, t) * exp(hat(dinit_(r, t)(w, v)))
    // where `w`, `v` are "small" elements of R^1 and R^2 respectively.
    //
    // Using `t * exp(hat(v)) == t + v` simplifies the equation to:
    //   init(r * exp(hat(w)), t + v) ~= init(r, t) * exp(hat(dinit_(r, t)(w, v)))
    //
    // Multiplying on the left by `init(r, t)^-1 == init(r^t, r^t * (-t))` gives
    //   exp(hat(dinit_(r, t)(w, v)))
    //     = init(r^t, r^t * (-t)) * init(r * exp(hat(w)), t + v)
    //     = init(exp(hat(w)), r^t * v)
    //
    // Since `w`, `v` are small, we can take logs of both sides to get
    //   dinit_(t, r)(w, v) = (w, r^t * v)
    //
    // Or in matrix notation:
    //   dinit_(t, r)(w, v) =  ( 1    0 ) ( w )
    //                         ( 0  r^t ) ( v )
    //
    // We actually need the pullback rather than the differential, so that's the transpose:
    //   pbinit_(t, r)(w, v) =  ( 1 0 ) ( w )
    //                          ( 0 r ) ( v )
    //
    // The pullback here implements the linear map corresponding to that matrix.
    (Pose2Coordinate(r, t), {
      let (w, v) = Self.decomposed(tangentVector: $0)
      return (w, r.rotate(v))
    })
  }

  /// The derivative of `t` satisfiying convention (2) described above.
  @derivative(of: Pose2Coordinate.t)
  @usableFromInline
  func vjpT() -> (value: Vector2, pullback: (Vector2) -> Vector3) {
    // Explanation of this calculation:
    //
    // `t` is the inverse of `init`, followed by a projection to the translation component. So we
    // can get the matrix for the pullback of `t` by inverting the matrix for the pullback of
    // `init` and then taking the relevant columns.
    //
    // The inverse is:
    //   ( 1    0 )
    //   ( 0  r^t )
    //
    // The pullback here implements the linear transformation represented by the first column of
    // this matrix.
    return (t, { Self.tangentVector(w: Vector1.zero, v: rot.unrotate($0)) } )
  }

  /// The derivative of `rot` satisfiying convention (2) described above.
  @derivative(of: Pose2Coordinate.rot)
  @usableFromInline
  func vjpRot() -> (value: Rot2, pullback: (Vector1) -> Vector3) {
    // This pullback is similar to the pullback in `vjpT`, but for the second column of the
    // matrix. See the comment in `vjpT` for more information.
    return (rot, { Self.tangentVector(w: $0, v: Vector2.zero) })
  }
}

// MARK: Coordinate Operators
public extension Pose2Coordinate {
  /// Product of two rotations.
  @differentiable
  static func * (lhs: Pose2Coordinate, rhs: Pose2Coordinate) -> Pose2Coordinate {
    Pose2Coordinate(lhs.rot * rhs.rot, lhs.t + lhs.rot * rhs.t)
  }

  /// Inverse of the rotation.
  @differentiable
  func inverse() -> Pose2Coordinate {
    Pose2Coordinate(self.rot.inverse(), self.rot.unrotate(-self.t))
  }
}

extension Pose2Coordinate: ManifoldCoordinate {
  public mutating func move(along direction: Vector3) {
    let (w, v) = Self.decomposed(tangentVector: direction)
    self.tStorage.move(along: rot.rotate(v))
    self.rotStorage.move(along: w)
  }
  
  /// p * Exp(q)
  @differentiable(wrt: local)
  public func global(_ local: Vector3) -> Self {
    self * Pose2Coordinate(Rot2(local.x), Vector2(local.y, local.z))
  }
  
  /// Log(p^{-1} * q)
  ///
  /// Explanation
  /// ====================
  /// `global_p(local_p(q)) = q`
  /// e.g. `p*Exp(Log(p^{-1} * q)) = q`
  /// QED
  ///
  @differentiable(wrt: global)
  public func local(_ global: Self) -> Vector3 {
    let d = self.inverse() * global
    return Vector3(d.rot.theta, d.t.x, d.t.y)
  }
}

/// Methods related to the Lie group structure.
extension Pose2 {
  /// The Adjoint group action of `self` on the tangent space, as a linear map.
  public var groupAdjoint: (Vector3) -> Vector3 {
    {
      let (w, v) = Pose2Coordinate.decomposed(tangentVector: $0)
      let tPerp = Vector2(-coordinate.t.y, coordinate.t.x)
      return Pose2Coordinate.tangentVector(w: w, v: coordinate.rot.rotate(v) - tPerp.scaled(by: w.x))
    }
  }

  /// The Adjoint group action of `self` on the tangent space, as a matrix.
  public var groupAdjointMatrix: Tensor<Double> {
    Tensor(stacking: Pose2.tangentStandardBasis.map { groupAdjoint($0).tensor }).transposed()
  }
}

extension Pose2: CustomStringConvertible {
  public var description: String {
    "Pose2(rot: \(coordinate.rot), t: \(coordinate.t))"
  }
}

/// Calculate relative pose 1T2 between two poses wT1 and wT2
@differentiable
public func between(_ wT1: Pose2, _ wT2: Pose2) -> Pose2 {
  wT1.inverse() * wT2
}
