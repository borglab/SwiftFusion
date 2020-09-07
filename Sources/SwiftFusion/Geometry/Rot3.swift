
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
  
  /// Create Manifold object from a quaternion representation
  /// Using the Eigen convention
  @differentiable
  public static func fromQuaternion(_ w: Double, _ x: Double, _ y: Double, _ z: Double) -> Self {
    let tx  = 2.0 * x
    let ty  = 2.0 * y
    let tz  = 2.0 * z
    let twx = tx * w
    let twy = ty * w
    let twz = tz * w
    let txx = tx * x
    let txy = ty * x
    let txz = tz * x
    let tyy = ty * y
    let tyz = tz * y
    let tzz = tz * z
    
    return Rot3(
      1.0-(tyy+tzz), txy-twz, txz+twy,
      txy+twz, 1.0-(txx+tzz), tyz-twx,
      txz-twy, tyz+twx, 1.0-(txx+tyy)
    )
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
  public static func * (r: Rot3, p: Vector3) -> Vector3 {
    r.rotate(p)
  }
}

/// determinant for a 3x3 matrix
fileprivate func det3(_ M: Tensor<Double>) -> Double {
  precondition(M.shape[0] == 3)
  precondition(M.shape[1] == 3)
  
  return (M[0, 0].scalar! * (M[1, 1].scalar! * M[2, 2].scalar! - M[2, 1].scalar! * M[1, 2].scalar!)
  - M[1, 0].scalar! * (M[0, 1].scalar! * M[2, 2].scalar! - M[2, 1].scalar! * M[0, 2].scalar!)
  + M[2, 0].scalar! * (M[0, 1].scalar! * M[1, 2].scalar! - M[1, 1].scalar! * M[0, 2].scalar!))
}

public extension Rot3 {
  /// Regularize a 3x3 matrix to find the closest rotation matrix in a Frobenius sense
  static func ClosestTo(mat: Matrix3) -> Rot3 {
    let M = Tensor<Double>(shape: [3, 3], scalars: [mat[0, 0], mat[0, 1], mat[0, 2], mat[1, 0], mat[1, 1], mat[1, 2], mat[2, 0], mat[2, 1], mat[2, 2]])
    
    let (_, U, V) = M.transposed().svd(computeUV: true, fullMatrices: true)
    let UVT = matmul(U!, V!.transposed())
    
    let det = det3(UVT)
    
    let S = Tensor<Double>(shape: [3], scalars: [1, 1, det]).diagonal()
    
    let R = matmul(V!, matmul(S, U!.transposed()))
    
    let M_r = Matrix3(R.scalars)
    
    return Rot3(coordinate: Matrix3Coordinate(M_r))
  }
}

public struct Matrix3Coordinate: Equatable, KeyPathIterable {
  public var R: Matrix3
  
  public typealias LocalCoordinate = Vector3
}

public extension Matrix3Coordinate {
  /// Construct from a rotation matrix, as doubles in *row-major* order
  @differentiable
  init(_ r11 : Double, _ r12 : Double, _ r13 : Double,
              _ r21 : Double, _ r22 : Double, _ r23 : Double,
              _ r31 : Double, _ r32 : Double, _ r33 : Double) {
    R = Matrix3(
      r11, r12, r13,
      r21, r22, r23,
      r31, r32, r33
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
  public static func * (lhs: Matrix3Coordinate, rhs: Matrix3Coordinate) -> Matrix3Coordinate {
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
    return matvec(R, v)
  }

  /// Returns the result of acting the inverse of `self` on `v`.
  @differentiable
  func unrotate(_ v: Vector3) -> Vector3 {
    return matvec(R.transposed(), v)
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
    let W = Matrix3(0.0, -wz, wy, wz, 0.0, -wx, -wy, wx, 0.0)
    let I_3x3 = Matrix3.identity
    if !nearZero {
      let theta = sqrtWrap(theta2)
      let sin_theta = sin(theta)
      let s2 = sin(theta / 2)
      let one_minus_cos = 2.0 * s2 * s2
      let K = W / theta
      let KK = matmul(K, K)
      
      return self * Matrix3Coordinate(
        I_3x3 + sin_theta * K
          + one_minus_cos * KK
      )
    } else {
      return self * Matrix3Coordinate(I_3x3 + W)
    }
  }

  @differentiable(wrt: global)
  public func localCoordinate(_ global: Matrix3Coordinate) -> Vector3 {
    let relative = self.inverse() * global
    let R = Vector9(relative.R)
    let (R11, R12, R13) = (R.s0, R.s1, R.s2)
    let (R21, R22, R23) = (R.s3, R.s4, R.s5)
    let (R31, R32, R33) = (R.s6, R.s7, R.s8)

    let tr = R11 + R22 + R33

    if tr + 1.0 < 1e-10 {
      if abs(R33 + 1.0) > 1e-5 {
        return (.pi / sqrtWrap(2.0 + 2.0 * R33)) * Vector3(R13, R23, 1.0 + R33)
      } else if abs(R22 + 1.0) > 1e-5 {
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
  init(_ matrix: Matrix3) {
    R = matrix
  }
}
