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
public struct Pose2: LieGroup, Equatable, KeyPathIterable {
  public typealias TangentVector = Vector3

  // MARK: - Manifold conformance

  public var coordinateStorage: Pose2Coordinate
  public init(coordinateStorage: Pose2Coordinate) { self.coordinateStorage = coordinateStorage }

  public mutating func move(along direction: Coordinate.LocalCoordinate) {
    coordinateStorage = coordinateStorage.retract(direction)
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
}

// MARK: Convenience initializers
extension Pose2 {
  /// Creates a `Pose2` with translation `x` and `y` and with rotation `theta`.
  @differentiable
  public init(_ x: Double, _ y: Double, _ theta: Double) {
    self.init(Rot2(theta), Vector2(x, y))
  }

  public init(
    randomWithCovariance covariance: Tensor<Double>,
    seed: TensorFlowSeed = Context.local.randomSeed
  ) {
    self.init(0, 0, 0)
    let r = matmul(cholesky(covariance), Tensor<Double>(randomNormal: [3, 1], seed: seed)).scalars
    let tv = Pose2.TangentVector(r[0], r[1], r[2])
    self.move(along: tv)
  }
}

extension Pose2 {
  /// Group action on `Vector2`.
  @differentiable
  public static func * (lhs: Pose2, rhs: Vector2) -> Vector2 {
    lhs.coordinate * rhs
  }
}

// MARK: - Global Coordinate System

public struct Pose2Coordinate: Equatable, KeyPathIterable {
  var t: Vector2
  var rot: Rot2
}

public extension Pose2Coordinate {
  @differentiable
  init(_ rot: Rot2, _ t: Vector2) {
    self.t = t
    self.rot = rot
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

// MARK: Coordinate Operators
extension Pose2Coordinate: LieGroupCoordinate {
  /// Creates the group identity.
  public init() {
    self.init(Rot2(), Vector2.zero)
  }

  /// Product of two transforms
  @differentiable
  public static func * (lhs: Pose2Coordinate, rhs: Pose2Coordinate) -> Pose2Coordinate {
    Pose2Coordinate(lhs.rot * rhs.rot, lhs * rhs.t)
  }

  /// Inverse of the rotation.
  @differentiable
  public func inverse() -> Pose2Coordinate {
    Pose2Coordinate(self.rot.inverse(), self.rot.unrotate(-self.t))
  }
}

extension Pose2Coordinate {
  /// Group action on `Vector2`.
  @differentiable
  public static func * (lhs: Pose2Coordinate, rhs: Vector2) -> Vector2 {
    lhs.t + lhs.rot * rhs
  }
}

extension Pose2Coordinate: ManifoldCoordinate {
  /// p * Exp(q)
  @differentiable(wrt: local)
  public func retract(_ local: Vector3) -> Self {
    // self * Pose2Coordinate(Rot2(local.x), Vector2(local.y, local.z))
    let v = Vector2(local.y,local.z)
    let w = local.x
    if (abs(w) < 1e-10) {
      return self * Pose2Coordinate(Rot2(w), v)
    } else {
      let R = Rot2(w)
      let v_ortho = Rot2(.pi/2) * v // points towards rot center
      let t_0 = (v_ortho - R.rotate(v_ortho))
      let t = Vector2(t_0.x / w, t_0.y / w)
      return self * Pose2Coordinate(R, t)
    }
  }
  
  /// Log(p^{-1} * q)
  ///
  /// Explanation
  /// ====================
  /// `global_p(local_p(q)) = q`
  /// e.g. `p*Exp(Log(p^{-1} * q)) = q`
  /// This invariant will be tested in the tests.
  ///
  @differentiable(wrt: global)
  public func localCoordinate(_ global: Self) -> Vector3 {
    let p = self.inverse() * global
//    return Vector3(d.rot.theta, d.t.x, d.t.y)
    let R = p.rot
    let t = p.t
    let w = R.theta
    if (abs(w) < 1e-10) {
      return Vector3(w, t.x, t.y);
    } else {
      let c_1 = R.c-1.0, s = R.s
      let det = c_1*c_1 + s*s
      let p = Rot2(.pi/2) * (R.unrotate(t) - t)
      let v = Vector2((w / det) * p.x, (w / det) * p.y)
      return Vector3(w, v.x, v.y)
    }
  }
}

/// Methods related to the Lie group structure.
extension Pose2Coordinate {
  public func Adjoint(_ v: Vector3) -> Vector3 {
    let (w, v) = Pose2Coordinate.decomposed(tangentVector: v)
    return Pose2Coordinate.tangentVector(
      w: w,
      v: rot.rotate(v) - w.x * Rot2(.pi / 2).rotate(t)
    )
  }

  public func AdjointTranspose(_ v: Vector3) -> Vector3 {
    let (w, v) = Pose2Coordinate.decomposed(tangentVector: v)
    return Pose2Coordinate.tangentVector(
      w: Vector1(w.x - t.x * v.y + t.y * v.x),
      v: rot.unrotate(v)
    )
  }
}

extension Pose2 {
  /// The Adjoint group action of `self` on the tangent space, as a matrix.
  public var AdjointMatrix: Tensor<Double> {
    Tensor(
      stacking: Pose2.TangentVector.standardBasis.map { Adjoint($0).flatTensor }).transposed()
  }
}

extension Pose2: CustomStringConvertible {
  public var description: String {
    "Pose2(rot: \(coordinate.rot), t: \(coordinate.t))"
  }
}
