import TensorFlow

/// SE(2) Lie group of 2D Euclidean Poses.
///
/// We adopt the conventions in [1] and [2] for the tangent space of this manifold and for
/// derivatives of functions to/from this manifold. In particular:
///
///   1. The tangent space at every point is the Lie algebra so(2).
///   2. For Lie groups `G`, `H`, with algebras `g`, `h`, the differential of any differentiable
///      function `f : G -> H` at `a \in G` is the linear map `df_a : g -> h` such that
///        f(a * exp(eps)) ~= f(a) * exp(df_a(eps))
///      where `eps` is a "small" element of `g`. (And the pullback is the dual map of `df_a`).
///
/// (1) is accomplished simply by defining the `TangentVector` type of `Pose2` to be so(2).
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
  /// The pose's translation.
  ///
  /// This is the projection SE(2) -> R^2. (Note: not a group homomorphism!)
  @differentiable public var t: Point2 { tStorage }

  /// The pose's rotation.
  ///
  /// This is the projection SE(2) -> SO(2).
  @differentiable public var rot: Rot2 { rotStorage }

  /// Creates a `Pose2` with rotation `r` and translation `t`.
  ///
  /// This is the bijection SO(2) x R^2 -> SE(2), where "x" means direct product of groups. (Note:
  /// not a group homomorphism!)
  @differentiable
  public init(_ t: Point2, _ r: Rot2) {
    tStorage = t
    rotStorage = r
  }

  /// The pullback of `init` satisfiying convention (2) described above.
  @derivative(of: init(_:_:))
  @usableFromInline
  static func vjpInit(_ t: Point2, _ r: Rot2)
    -> (value: Pose2, pullback: (TangentVector) -> (Point2.TangentVector, Rot2.TangentVector))
  {
    // Explanation of this calculation:
    //
    // We would like the differential of `init` at `(t, r)`, `dinit_(t, r)`, to satisfy:
    //   init(t * exp(eps_t), r * exp(eps_r)) ~= init(t, r) * exp(dinit_(t, r)(eps_t, eps_r))
    // where `eps_t`, `eps_r` are "small" elements of R^2 and so(2) respectively.
    //
    // Multiplying on the left by `init(t, r)^-1` gives
    //   exp(dinit_(t, r)(eps_t, eps_r))
    //     = init(t, r)^-1 * init(t * exp(eps_t), r * exp(eps_r))
    //     = init(r^-1 * (-t), inverse(r)) * init(t * exp(eps_t), r * exp(eps_r))
    //     = init(r^-1 * exp(eps_t), exp(eps_r))
    //
    // Since `eps_t`, `eps_r` are small, we can take logs of both sides to get
    //   dinit_(t, r)(eps_t, eps_r) = (r^-1 * eps_t, eps_r)
    //
    // Or in matrix notation:
    //   dinit_(t, r)(eps_t, eps_r) =  ( r^-1  0 ) ( eps_t )
    //                                 ( 0     1 ) ( eps_r )
    //
    // We actually need the pullback rather than the differential, so that's the transpose:
    //   pbinit_(t, r)(eps_t, eps_r) =  ( r 0 ) ( eps_t )
    //                                  ( 0 1 ) ( eps_r )
    //
    // The pullback here implements the linear map corresponding to that matrix.
    (Pose2(t, r), { eps in
      (
        Point2.TangentVector(
          x: r.c * eps.vx - r.s * eps.vy,
          y: r.s * eps.vx + r.c * eps.vy
        ),
        eps.omega
      )
    })
  }

  /// The pullback of `t` satisfiying convention (2) described above.
  @derivative(of: t)
  @usableFromInline
  func vjpT() -> (value: Point2, pullback: (Point2.TangentVector) -> TangentVector) {
    // Explanation of this calculation:
    //
    // `t` is the inverse of `init`, followed by a projection to the translation component. So we
    // can get the matrix for the pullback of `t` by inverting then taking the relevant columns.
    //
    // The inverse is:
    //   ( r^-1 0 )
    //   ( 0    1 )
    //
    // The below implements the linear transformation represented by the first column of this matrix.
    return (t, { v in
      TangentVector(
        vx: rot.c * v.x + rot.s * v.y,
        vy: -rot.s * v.x + rot.c * v.y,
        omega: .zero
      )
    })
  }

  /// The pullback of `t` satisfiying convention (2) described above.
  @derivative(of: rot)
  @usableFromInline
  func vjpRot() -> (value: Rot2, pullback: (Rot2.TangentVector) -> TangentVector) {
    // This pullback is similar to the pullback in `vjpT`, but for the second column of the
    // matrix. See the comment in `vjpT` for more information.
    return (rot, { TangentVector(vx: 0, vy: 0, omega: $0) })
  }

  /// The Lie algebra se(2).
  public struct TangentVector:
    Differentiable, AdditiveArithmetic, KeyPathIterable, VectorProtocol, ElementaryFunctions
  {
    public typealias VectorSpaceScalar = Double

    /// The x translation component in the basis defined in section 4.1 of
    /// https://github.com/borglab/gtsam/blob/develop/doc/LieGroups.pdf.
    public var vx: Double

    /// The y translation component in the basis defined in section 4.1 of
    /// https://github.com/borglab/gtsam/blob/develop/doc/LieGroups.pdf.
    public var vy: Double

    /// The rotation component in the basis defined in section 4.1 of
    /// https://github.com/borglab/gtsam/blob/develop/doc/LieGroups.pdf.
    public var omega: Double

    public init(_ vx: Double, _ vy: Double, _ omega: Double) {
      self.vx = vx
      self.vy = vy
      self.omega = omega
    }
  }

  /// Moves `self` by `exp(direction)`.
  public mutating func move(along direction: TangentVector) {
    // TODO: This should be the real exponential map.
    self.tStorage.move(along: Point2.TangentVector(
      x: rot.c * direction.vx - rot.s * direction.vy,
      y: rot.s * direction.vx + rot.c * direction.vy
    ))
    self.rotStorage.move(along: direction.omega)
  }

  /// Storage for the pose's translation.
  ///
  /// This is private so that we can define an accessor with a custom derivative (Swift AD does not
  /// currently support custom derivatives on stored properties).
  private var tStorage: Point2

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
    self.init(Point2(x, y), Rot2(theta))
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
    Pose2(a.t + a.rot * b.t, a.rot * b.rot)
  }

  /// The adjoint representation of `self`, as a linear map.
  public func adjoint(_ v: TangentVector) -> TangentVector {
    TangentVector(
      rot.c * v.vx - rot.s * v.vy + t.y * v.omega,
      rot.s * v.vx + rot.c * v.vy - t.x * v.omega,
      v.omega
    )
  }

  /// The adjoint representation of `self`, as a matrix.
  public var adjointMatrix: Tensor<Double> {
    Tensor(matrixRows: Pose2.tangentStandardBasis.map { adjoint($0) }).transposed()
  }
}

/// Group inverse.
@differentiable
public func inverse(_ p: Pose2) -> Pose2 {
  Pose2(p.rot.unrotate(-p.t), inverse(p.rot))
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
