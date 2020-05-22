import TensorFlow

/// SE(3) Lie group of 3D Euclidean Poses.
public struct Pose3: LieGroup, Equatable, KeyPathIterable {
  // MARK: - Manifold conformance

  public typealias Coordinate = Pose3Coordinate
  public typealias TangentVector = Vector6
  
  public var coordinateStorage: Pose3Coordinate
  public init(coordinateStorage: Pose3Coordinate) { self.coordinateStorage = coordinateStorage }

  public mutating func move(along direction: Coordinate.LocalCoordinate) {
    coordinateStorage = coordinateStorage.retract(direction)
  }

  /// Creates a `Pose3` with rotation `r` and translation `t`.
  ///
  /// This is the bijection SO(3) x R^3 -> SE(3), where "x" means direct product of groups.
  @differentiable
  public init(_ r: Rot3, _ t: Vector3) {
    self.init(coordinate: Pose3Coordinate(r, t))
  }
  
  // MARK: Convenience Attributes
  
  @differentiable public var t: Vector3 { coordinate.t }
  
  @differentiable public var rot: Rot3 { coordinate.rot }
  
  /// Create from an element in tangent space (Expmap)
  @differentiable
  public static func fromTangent(_ vector: Vector6) -> Self {
    return Pose3(coordinate: Pose3Coordinate(Rot3(), Vector3.zero).retract(vector))
  }
}

// MARK: - Global Coordinate System

public struct Pose3Coordinate: Equatable, KeyPathIterable {
  var rot: Rot3
  var t: Vector3
  
  public typealias LocalCoordinate = Vector6
}

public extension Pose3Coordinate {
  @differentiable
  init(_ rot: Rot3, _ t: Vector3) {
    self.rot = rot
    self.t = t
  }

  fileprivate struct DecomposedTangentVector: Differentiable {
    /// Rotation component.
    var w: Vector3

    /// Translation component.
    var v: Vector3
  }

  /// Returns the components of `tangentVector`.
  @differentiable
  fileprivate static func decomposed(tangentVector: Vector6) -> DecomposedTangentVector {
    return DecomposedTangentVector(
      w: Vector3(tangentVector.s0, tangentVector.s1, tangentVector.s2),
      v: Vector3(tangentVector.s3, tangentVector.s4, tangentVector.s5)
    )
  }
  
  /// Creates a tangent vector given its components.
  @differentiable
  fileprivate static func tangentVector(_ t: DecomposedTangentVector) -> Vector6 {
    let w = t.w
    let v = t.v
    return Vector6(w.x, w.y, w.z, v.x, v.y, v.z)
  }
}

// MARK: Coordinate Operators
extension Pose3Coordinate: LieGroupCoordinate {
  /// Creates the group identity.
  public init() {
    self.init(Rot3(), Vector3.zero)
  }

  /// Product of two transforms
  @differentiable
  public static func * (lhs: Pose3Coordinate, rhs: Pose3Coordinate) -> Pose3Coordinate {
    Pose3Coordinate(lhs.rot * rhs.rot, lhs.t + lhs.rot * rhs.t)
  }

  /// Inverse of the rotation.
  @differentiable
  public func inverse() -> Pose3Coordinate {
    Pose3Coordinate(self.rot.inverse(), self.rot.unrotate(-self.t))
  }
}

extension Vector3 {
  public func cross(_ b: Vector3) -> Vector3 {
    let a = self
    return Vector3(
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x)
  }
}

@differentiable
public func skew_symmetric_v(_ v: Vector3) -> Tensor<Double> {
  Tensor<Double>(shape: [3, 3], scalars: [
        0, -v.z, v.y,
        v.z, 0, -v.x,
        -v.y, v.x, 0]
    )
}

extension Pose3Coordinate: ManifoldCoordinate {
  /// p * Exp(q)
  @differentiable(wrt: local)
  public func retract(_ local: Vector6) -> Self {
    // get angular velocity omega and translational velocity v from twist xi
    let decomposed = Pose3Coordinate.decomposed(tangentVector: local)
    let omega = decomposed.w
    let v = decomposed.v
    
    let R = Rot3.fromTangent(omega)
    
    let theta2 = omega.squaredNorm
    if theta2 > .ulpOfOne {
        let t_parallel = omega.dot(v) * omega // translation parallel to axis
        let omega_cross_v = omega.cross(v); // points towards axis
        let t = (1 / theta2) * (omega_cross_v - R * omega_cross_v + t_parallel as Vector3)
        return self * Pose3Coordinate(R, t)
    } else {
        return self * Pose3Coordinate(R, v)
    }
  }
  
  /// Log(p^{-1} * q)
  ///
  /// Explanation
  /// ====================
  /// `global_p(local_p(q)) = q`
  /// e.g. `p*Exp(Log(p^{-1} * q)) = q`
  /// This invariant will be tested in the tests.
  @differentiable(wrt: global)
  public func localCoordinate(_ global: Self) -> Vector6 {
    let relative = self.inverse() * global
    let w = Rot3().coordinate.localCoordinate(relative.rot.coordinate)
    // TODO(SR-12776): The `localSmallRot` and `localBigRot` helpers work around this bug. Once it
    // is fixed, we can inline them.
    if w.norm < 1e-10 {
      return localSmallRot(global)
    } else {
      return localBigRot(global)
    }
  }

  /// Implements `local` in the small rotation case.
  @differentiable(wrt: global)
  private func localSmallRot(_ global: Self) -> Vector6 {
    let relative = self.inverse() * global
    let w = Rot3().coordinate.localCoordinate(relative.rot.coordinate)
    let T = relative.t
    return Pose3Coordinate.tangentVector(DecomposedTangentVector(w: w, v: T))
  }

  /// Implements `local` in the non-small rotation case.
  @differentiable(wrt: global)
  private func localBigRot(_ global: Self) -> Vector6 {
    let relative = self.inverse() * global
    let w = Rot3().coordinate.localCoordinate(relative.rot.coordinate)
    let T = relative.t.tensor.reshaped(to: [3, 1])
    let t = w.norm
    let W = skew_symmetric_v((1 / t) * w);
    // Formula from Agrawal06iros, equation (14)
    // simplified with Mathematica, and multiplying in T to avoid matrix math
    let Tan = tan(0.5 * t)
    let WT = matmul(W, T)
    let u = ((T - (0.5 * t) * WT) as Tensor<Double>) + (1 - t / (2 * Tan)) * matmul(W, WT)
    precondition(u.shape == [3, 1])
    return Pose3Coordinate.tangentVector(
      DecomposedTangentVector(w: w, v: Vector3(u.reshaped(to: [3]))))
  }

}

extension Pose3: CustomStringConvertible {
  public var description: String {
    "Pose3(rot: \(coordinate.rot), t: \(coordinate.t))"
  }
}
