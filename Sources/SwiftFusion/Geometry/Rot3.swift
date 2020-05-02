
import TensorFlow

extension Vector3 {
  static func * (_ s:Double, _ vector:Vector3) -> Vector3 {
    Vector3(s * vector.x, s * vector.y, s * vector.z)
  }
}

public struct Rot3: Manifold, TangentStandardBasis, Equatable, KeyPathIterable {
  public typealias TangentVector = Vector3
  // MARK: - Manifold conformance

  public var coordinateStorage: Matrix3Coordinate
  public init(coordinateStorage: Matrix3Coordinate) { self.coordinateStorage = coordinateStorage }

  public mutating func move(along direction: Coordinate.LocalCoordinate) {
    coordinateStorage = coordinateStorage.global(direction)
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
  
  public static func fromTangent(_ vector: Vector3) -> Self {
    var actual = Rot3()
    actual.move(along: vector)
    return actual
  }
  
  public init() {
    self.init(coordinateStorage: Matrix3Coordinate(eye(rowCount: 3)))
  }
}

public struct Matrix3Coordinate: Equatable, KeyPathIterable {
  public var R: Tensor<Double>
}

public extension Matrix3Coordinate {
  /// Construct from a rotation matrix, as doubles in *row-major* order
  @differentiable
  init(_ r11 : Double, _ r12 : Double, _ r13 : Double,
              _ r21 : Double, _ r22 : Double, _ r23 : Double,
              _ r31 : Double, _ r32 : Double, _ r33 : Double) {
    R = Tensor<Double>(shape: [3,3], scalars: [r11, r12, r13,
                                               r21, r22, r23,
                                               r31, r32, r33])
  }
  
  /// Product of two rotations.
  @differentiable(wrt: (lhs, rhs))
  static func * (lhs: Matrix3Coordinate, rhs: Matrix3Coordinate) -> Matrix3Coordinate {
    Matrix3Coordinate(matmul(lhs.R, rhs.R))
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
  public func global(_ local: Vector3) -> Matrix3Coordinate {
    let theta2 = local.squaredNorm
    let nearZero = theta2 <= .ulpOfOne
    let (wx, wy, wz) = (local.x, local.y, local.z)
    let W = Tensor<Double>(shape: [3, 3], scalars: [0.0, -wz, wy, wz, 0.0, -wx, -wy, wx, 0.0])
    let I_3x3: Tensor<Double> = eye(rowCount: 3)
    if !nearZero {
      let theta = sqrtWrap(theta2)
      let sin_theta = sin(theta)
      let s2 = sin(theta / 2)
      let one_minus_cos = 2.0 * s2 * s2
      let K = W / theta
      let KK = matmul(K, K)
      
      return self * Matrix3Coordinate(
        I_3x3 + Tensor<Double>(repeating: sin_theta, shape: [3, 3]) * K
          + Tensor<Double>(repeating: one_minus_cos, shape: [3, 3]) * KK
      )
    } else {
      return self * Matrix3Coordinate(I_3x3 + W)
    }
  }

  @differentiable(wrt: global)
  public func local(_ global: Matrix3Coordinate) -> Vector3 {
    let (R11, R12, R13) = (R[0, 0].scalar!, R[0, 1].scalar!, R[0, 2].scalar!)
    let (R21, R22, R23) = (R[1, 0].scalar!, R[1, 1].scalar!, R[1, 2].scalar!)
    let (R31, R32, R33) = (R[2, 0].scalar!, R[2, 1].scalar!, R[2, 2].scalar!)

    let tr = R.diagonalPart().sum().scalar!

    var omega: Vector3
    
    print("trace = \(tr)")
    if abs(tr + 1.0) < 1e-10 {
      if abs(R33 + 1.0) > 1e-10 {
        omega = (.pi / sqrtWrap(2.0 + 2.0 * R33)) * Vector3(R13, R23, 1.0 + R33)
      } else if abs(R22 + 1.0) > 1e-10 {
        omega = (.pi / sqrtWrap(2.0 + 2.0 * R22)) * Vector3(R12, 1.0 + R22, R32)
      } else {
        // if(abs(R.r1_.x()+1.0) > 1e-10)  This is implicit
        omega = (.pi / sqrtWrap(2.0 + 2.0 * R11)) * Vector3(1.0 + R11, R21, R31)
      }
    } else {
      var magnitude: Double

      let tr_3 = tr - 3.0; // always negative
      if tr_3 < -1e-7 {
        let theta = acos((tr - 1.0) / 2.0)
        magnitude = theta / (2.0 * sin(theta))
      } else {
        // when theta near 0, +-2pi, +-4pi, etc. (trace near 3.0)
        // use Taylor expansion: theta \approx 1/2-(t-3)/12 + O((t-3)^2)
        magnitude = 0.5 - tr_3 * tr_3 / 12.0;
      }
      omega = magnitude * Vector3(R32 - R23, R13 - R31, R21 - R12);
    }
    return omega
  }

  /// Construct from Tensor
  init(_ tensor: Tensor<Double>) {
    R = tensor
  }
}
