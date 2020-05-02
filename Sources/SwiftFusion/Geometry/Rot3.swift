
import TensorFlow

public struct Rot3: Manifold, TangentStandardBasis, Equatable, KeyPathIterable {
  public typealias TangentVector = Vector3
  // MARK: - Manifold conformance

  public var coordinateStorage: Matrix3Coordinate
  public init(coordinateStorage: Matrix3Coordinate) { self.coordinateStorage = coordinateStorage }

  public mutating func move(along direction: Coordinate.LocalCoordinate) {
    coordinateStorage = coordinateStorage.global(direction)
  }

// BUG(fan): will trigger ICE
//  /// Construct from a rotation matrix, as doubles in *row-major* order
//  @differentiable
//  public init(_ r11 : Double, _ r12 : Double, _ r13 : Double,
//              _ r21 : Double, _ r22 : Double, _ r23 : Double,
//              _ r31 : Double, _ r32 : Double, _ r33 : Double) {
//    self.init(coordinateStorage: Matrix3Coordinate(r11, r12, r13,
//              r21, r22, r23,
//              r31, r32, r33))
//  }
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
    Vector3(1,2,3)
  }

  /// Construct from Tensor
  init(_ tensor: Tensor<Double>) {
    R = tensor
  }
}
