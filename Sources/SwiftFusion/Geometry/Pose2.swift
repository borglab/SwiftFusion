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
public struct Pose2: Differentiable, TangentStandardBasis, KeyPathIterable {
  /// The pose's rotation.
  ///
  /// This is the projection SE(2) -> SO(2).
  @differentiable public var rot: Rot2 { rotStorage }

  /// The pose's translation.
  ///
  /// This is the projection SE(2) -> R^2. (Note: not a group homomorphism!)
  @differentiable public var t: Vector2 { tStorage }

  /// Creates a `Pose2` with rotation `r` and translation `t`.
  ///
  /// This is the bijection SO(2) x R^2 -> SE(2), where "x" means direct product of groups. (Note:
  /// not a group homomorphism!)
  @differentiable
  public init(_ r: Rot2, _ t: Vector2) {
    tStorage = t
    rotStorage = r
  }

  /// The derivative of `init` satisfiying convention (2) described above.
  @derivative(of: init(_:_:))
  @usableFromInline
  static func vjpInit(_ r: Rot2, _ t: Vector2)
    -> (value: Pose2, pullback: (TangentVector) -> (Rot2.TangentVector, Vector2))
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
    (Pose2(r, t), {
      let (w, v) = Self.decomposed(tangentVector: $0)
      return (w, r.rotate(v))
    })
  }

  /// The derivative of `t` satisfiying convention (2) described above.
  @derivative(of: t)
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
  @derivative(of: rot)
  @usableFromInline
  func vjpRot() -> (value: Rot2, pullback: (Vector1) -> Vector3) {
    // This pullback is similar to the pullback in `vjpT`, but for the second column of the
    // matrix. See the comment in `vjpT` for more information.
    return (rot, { Self.tangentVector(w: $0, v: Vector2.zero) })
  }

  /// Twist coordinates, with the first component for rotation and the remaining components for
  /// translation.
  ///
  /// See the documentation comment on the declaration of `Pose2` for more information.
  public typealias TangentVector = Vector3

  /// Creates a tangent vector given rotation (`w`) and translation (`v`) components.
  public static func tangentVector(w: Vector1, v: Vector2) -> Vector3 {
    Vector3(w.x, v.x, v.y)
  }

  /// Returns the rotation (`w`) and translation (`v`) components of `tangentVector`.
  public static func decomposed(tangentVector: Vector3) -> (w: Vector1, v: Vector2) {
    (Vector1(tangentVector.x), Vector2(tangentVector.y, tangentVector.z))
  }

  /// Moves `self` by `exp(hat(direction))`.
  public mutating func move(along direction: Vector3) {
    // TODO: This should be the real exponential map.
    let (w, v) = Self.decomposed(tangentVector: direction)
    self.tStorage.move(along: rot.rotate(v))
    self.rotStorage.move(along: w)
  }

  /// Storage for the pose's translation.
  ///
  /// This is private so that we can define an accessor with a custom derivative (Swift AD does not
  /// currently support custom derivatives on stored properties).
  private var tStorage: Vector2

  /// Storage for the pose's rotation.
  ///
  /// This is private so that we can define an accessor with a custom derivative (Swift AD does not
  /// currently support custom derivatives on stored properties).
  private var rotStorage: Rot2
}

/// Convenience initializers.
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

/// Methods related to the Lie group structure.
extension Pose2 {
  /// Group operation.
  @differentiable
  public static func * (a: Pose2, b: Pose2) -> Pose2 {
    Pose2(a.rot * b.rot, a.t + a.rot * b.t)
  }

  /// The Adjoint group action of `self` on the tangent space, as a linear map.
  public var groupAdjoint: (Vector3) -> Vector3 {
    {
      let (w, v) = Self.decomposed(tangentVector: $0)
      let tPerp = Vector2(-t.y, t.x)
      return Self.tangentVector(w: w, v: rot.rotate(v) - tPerp.scaled(by: w.x))
    }
  }

  /// The Adjoint group action of `self` on the tangent space, as a matrix.
  public var groupAdjointMatrix: Tensor<Double> {
    Tensor(stacking: Pose2.tangentStandardBasis.map { groupAdjoint($0).tensor }).transposed()
  }
}

/// Group inverse.
@differentiable
public func inverse(_ p: Pose2) -> Pose2 {
  Pose2(inverse(p.rot), p.rot.unrotate(-p.t))
}

extension Pose2: Equatable {
  public static func == (lhs: Pose2, rhs: Pose2) -> Bool {
    (lhs.t, lhs.rot) == (rhs.t, rhs.rot)
  }
}

/// Calculate relative pose 1T2 between two poses wT1 and wT2
@differentiable
public func between(_ wT1: Pose2, _ wT2: Pose2) -> Pose2 {
  inverse(wT1) * wT2
}
