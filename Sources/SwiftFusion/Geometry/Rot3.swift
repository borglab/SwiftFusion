
import TensorFlow

/// SO(3) group of 3D Rotations
public struct Rot3: LieGroup, Equatable, KeyPathIterable {
  public typealias TangentVector = Vector3
  // MARK: - Manifold conformance

  public var coordinateStorage: Matrix3Coordinate
  public init(coordinateStorage: Matrix3Coordinate) { self.coordinateStorage = coordinateStorage }

  public mutating func move(along direction: Coordinate.LocalCoordinate) {
    coordinateStorage = coordinateStorage.retract(direction)
  }

  /// Construct from a rotation matrix, as doubles in *row-major* order
  @differentiable
  public init(_ r11 : Double, _ r12 : Double, _ r13 : Double,
              _ r21 : Double, _ r22 : Double, _ r23 : Double,
              _ r31 : Double, _ r32 : Double, _ r33 : Double) {
    self.init(coordinate: Matrix3Coordinate(r11, r12, r13,
              r21, r22, r23,
              r31, r32, r33))
  }
  
  /// Create Manifold object from element of the tangent (Expmap)
  @differentiable
  public static func fromTangent(_ vector: Vector3) -> Self {
    return Rot3(coordinate: Rot3().coordinate.retract(vector))
  }
  
  /// Returns the result of acting `self` on `v`.
  @differentiable
  func rotate(_ v: Vector3) -> Vector3 {
    return coordinate.rotate(v)
  }
  
  /// Returns the result of acting the inverse of `self` on `v`.
  @differentiable
  func unrotate(_ v: Vector3) -> Vector3 {
    return coordinate.unrotate(v)
  }
  
  /// Returns the result of acting `self` on `v`.
  @differentiable
  static func *^ (r: Rot3, p: Vector3) -> Vector3 {
    r.rotate(p)
  }
}

public struct Matrix3Coordinate: Equatable, KeyPathIterable {
  public var R: Tensor<Double>
  
  public typealias LocalCoordinate = Vector3
}

public extension Matrix3Coordinate {
  /// Construct from a rotation matrix, as doubles in *row-major* order
  @differentiable
  init(_ r11 : Double, _ r12 : Double, _ r13 : Double,
              _ r21 : Double, _ r22 : Double, _ r23 : Double,
              _ r31 : Double, _ r32 : Double, _ r33 : Double) {
    R = matrixTensor(
      r11, r12, r13,
      r21, r22, r23,
      r31, r32, r33
    )
  }

  /// Derivative of the above `init`.
  @derivative(of: init)
  static func vjpInit(
    _ r11 : Double, _ r12 : Double, _ r13 : Double,
    _ r21 : Double, _ r22 : Double, _ r23 : Double,
    _ r31 : Double, _ r32 : Double, _ r33 : Double
  ) -> (
    value: Matrix3Coordinate,
    pullback: (TangentVector) -> (
      Double, Double, Double,
      Double, Double, Double,
      Double, Double, Double
    )
  ) {
    func pullback(_ v: TangentVector) -> (
      Double, Double, Double,
      Double, Double, Double,
      Double, Double, Double
    ) {
      let s = v.R.scalars
      return (
        s[0], s[1], s[2],
        s[3], s[4], s[5],
        s[6], s[7], s[8]
      )
    }
    return (
      Matrix3Coordinate(
        r11, r12, r13,
        r21, r22, r23,
        r31, r32, r33
      ),
      pullback
    )
  }
}

extension Matrix3Coordinate: LieGroupCoordinate {
  /// Creates the group identity.
  public init() {
    self.init(
      1, 0, 0,
      0, 1, 0,
      0, 0, 1
    )
  }
  
  /// Product of two rotations.
  @differentiable(wrt: (lhs, rhs))
  public static func ** (lhs: Matrix3Coordinate, rhs: Matrix3Coordinate) -> Matrix3Coordinate {
    Matrix3Coordinate(matmul(lhs.R, rhs.R))
  }
  
  /// Inverse of the rotation.
  @differentiable
  public func inverse() -> Matrix3Coordinate {
    Matrix3Coordinate(R.transposed())
  }

  @differentiable(wrt: v)
  public func Adjoint(_ v: Vector3) -> Vector3 {
    return rotate(v)
  }

  @differentiable(wrt: v)
  public func AdjointTranspose(_ v: Vector3) -> Vector3 {
    return unrotate(v)
  }
}

/// Actions.
extension Matrix3Coordinate {
  /// Returns the result of acting `self` on `v`.
  @differentiable
  func rotate(_ v: Vector3) -> Vector3 {
    return Vector3(matmul(R, v.tensor.reshaped(to: [3, 1])).reshaped(to: [3]))
  }

  /// Returns the result of acting the inverse of `self` on `v`.
  @differentiable
  func unrotate(_ v: Vector3) -> Vector3 {
    return Vector3(matmul(R, transposed: true, v.tensor.reshaped(to: [3, 1])).reshaped(to: [3]))
  }
}

func sqrtWrap(_ v: Double) -> Double {
  sqrt(v)
}

@derivative(of: sqrtWrap)
func _vjpSqrt(_ v: Double) -> (value: Double, pullback: (Double) -> Double) {
  let r = sqrt(v)
  return (r, {
    1/(2*r) * $0
  })
}

extension Matrix3Coordinate: ManifoldCoordinate {
  /// Compose with the exponential map
  @differentiable(wrt: local)
  public func retract(_ local: Vector3) -> Matrix3Coordinate {
    let theta2 = local.squaredNorm
    let nearZero = theta2 <= .ulpOfOne
    let (wx, wy, wz) = (local.x, local.y, local.z)
    let W = matrixTensor(0.0, -wz, wy, wz, 0.0, -wx, -wy, wx, 0.0)
    let I_3x3: Tensor<Double> = eye(rowCount: 3)
    if !nearZero {
      let theta = sqrtWrap(theta2)
      let sin_theta = sin(theta)
      let s2 = sin(theta / 2)
      let one_minus_cos = 2.0 * s2 * s2
      let K = W / theta
      let KK = matmul(K, K)
      
      return self ** Matrix3Coordinate(
        I_3x3 + Tensor<Double>(repeating: sin_theta, shape: [3, 3]) * K
          + Tensor<Double>(repeating: one_minus_cos, shape: [3, 3]) * KK
      )
    } else {
      return self ** Matrix3Coordinate(I_3x3 + W)
    }
  }

  @differentiable(wrt: global)
  public func localCoordinate(_ global: Matrix3Coordinate) -> Vector3 {
    let relative = self.inverse() ** global
    let R = relative.R
    let (R11, R12, R13) = (R[0, 0].scalars[0], R[0, 1].scalars[0], R[0, 2].scalars[0])
    let (R21, R22, R23) = (R[1, 0].scalars[0], R[1, 1].scalars[0], R[1, 2].scalars[0])
    let (R31, R32, R33) = (R[2, 0].scalars[0], R[2, 1].scalars[0], R[2, 2].scalars[0])

    let tr = R.diagonalPart().sum().scalars[0]

    if abs(tr + 1.0) < 1e-10 {
      if abs(R33 + 1.0) > 1e-10 {
        return (.pi / sqrtWrap(2.0 + 2.0 * R33)) * Vector3(R13, R23, 1.0 + R33)
      } else if abs(R22 + 1.0) > 1e-10 {
        return (.pi / sqrtWrap(2.0 + 2.0 * R22)) * Vector3(R12, 1.0 + R22, R32)
      } else {
        // if(abs(R.r1_.x()+1.0) > 1e-10)  This is implicit
        return (.pi / sqrtWrap(2.0 + 2.0 * R11)) * Vector3(1.0 + R11, R21, R31)
      }
    } else {
      let tr_3 = tr - 3.0; // always negative
      if tr_3 < -1e-7 {
        let theta = acos((tr - 1.0) / 2.0)
        let magnitude = theta / (2.0 * sin(theta))
        return magnitude * Vector3(R32 - R23, R13 - R31, R21 - R12)
      } else {
        // when theta near 0, +-2pi, +-4pi, etc. (trace near 3.0)
        // use Taylor expansion: theta \approx 1/2-(t-3)/12 + O((t-3)^2)
        let magnitude = (0.5 - tr_3 * tr_3 / 12.0)
        return magnitude * Vector3(R32 - R23, R13 - R31, R21 - R12)
      }
    }
  }

  /// Construct from Tensor
  init(_ tensor: Tensor<Double>) {
    R = tensor
  }
}

/// Returns a matrix tensor containing the given scalars.
// TODO: This is a workaround for the problem mentioned in
// https://github.com/apple/swift/pull/31723. When that fix is available, we can delete the
// custom derivative of this function, and inline the function into its callsites.
@differentiable
fileprivate func matrixTensor(
  _ r11 : Double, _ r12 : Double, _ r13 : Double,
  _ r21 : Double, _ r22 : Double, _ r23 : Double,
  _ r31 : Double, _ r32 : Double, _ r33 : Double
) -> Tensor<Double> {
  return Tensor<Double>(shape: [3,3], scalars: [r11, r12, r13,
                                                r21, r22, r23,
                                                r31, r32, r33])
}

/// Derivative of `matrixTensor`.
// This works around a problem with differentiating array literals:
// https://github.com/apple/swift/pull/31723
@derivative(of: matrixTensor)
fileprivate func vjpMatrixTensor(
  _ r11 : Double, _ r12 : Double, _ r13 : Double,
  _ r21 : Double, _ r22 : Double, _ r23 : Double,
  _ r31 : Double, _ r32 : Double, _ r33 : Double
) -> (
  value: Tensor<Double>,
  pullback: (Tensor<Double>) -> (
    Double, Double, Double,
    Double, Double, Double,
    Double, Double, Double
  )
) {
  func pullback(_ v: Tensor<Double>) -> (
    Double, Double, Double,
    Double, Double, Double,
    Double, Double, Double
  ) {
    let s = v.scalars
    return (
      s[0], s[1], s[2],
      s[3], s[4], s[5],
      s[6], s[7], s[8]
    )
  }
  return (
    matrixTensor(
      r11, r12, r13,
      r21, r22, r23,
      r31, r32, r33
    ),
    pullback
  )
}
