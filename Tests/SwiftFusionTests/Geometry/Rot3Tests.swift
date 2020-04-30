// Unit tests for Rot3 class, which models SO(3)

import SwiftFusion

import TensorFlow

/// Matrix-vector multiply
extension Vector3 {
  static func * (_ tensor:Tensor<Double>, _ vector:Vector3) -> Vector3 {
    Vector3(matmul(tensor, vector.tensor))
  }
}

/// scalar-vector multiply
extension Vector3 {
  static func * (_ s:Double, _ vector:Vector3) -> Vector3 {
    Vector3(s * vector.x, s * vector.y, s * vector.z)
  }
}

/// Functor implementing Exponential map and its derivative/pullback
public struct ExpmapFunctor {
  let theta : Double
  let theta2 : Double
  let nearZero : Bool
  let W : Tensor<Double>
  // below only defined if !nearZero:
  let sin_theta, one_minus_cos : Double?
  let K, KK : Tensor<Double>?

  /// Constructor with element of Lie algebra so(3)
  public init(_ omega : Vector3) {
    theta2 = omega.squaredNorm
    theta = theta2.squareRoot()
    let wx = omega.x, wy = omega.y, wz = omega.z
    W = Tensor<Double>(shape: [3,3], scalars: [0.0, -wz, wy, wz, 0.0, -wx, -wy, wx, 0.0])
    nearZero = theta2 <= 1e-9
    // Define quantities only needed far from zero below
    if (nearZero) {
      sin_theta = nil
      one_minus_cos = nil
      K = nil
      KK = nil
    }
  else {
    sin_theta = .some(sin(theta))
    let s2 = sin(theta / 2.0)
    one_minus_cos = .some(2.0 * s2 * s2)  // numerically better than [1 - cos(theta)]
    K = W / theta
    KK = matmul(K!,K!)
  }
}

  /// Rodrigues formula
  func expmap() -> Tensor<Double> {
    if (nearZero) {
      return eye(rowCount:3) + W
    }
    else {
      return eye(rowCount:3) + sin_theta! * K! + one_minus_cos! * KK!
    }
  }

  /// Apply derivative of exponential map
  /// Rot3.TangentVector = dexp() * Vector3.TangentVector
  /// Currently unused, as we use pullbackDexp below.
  func applyDexp(_ delta : Vector3) -> Matrix3Coordinate.TangentVector {
    if (nearZero) {
      return delta - 0.5 * (W * delta)
    } else {
      let a = one_minus_cos! / theta
      let b = 1.0 - sin_theta! / theta
      return delta - a * (K! * delta) + b * (KK! * delta)
    }
  }

  /// Pullback of exponential map
  /// Vector3.TangentVector = dexp()^T * Rot3.TangentVector
  func pullbackDexp(_ delta : Matrix3Coordinate.TangentVector) -> Vector3 {
    if (nearZero) {
      return delta + 0.5 * (W * delta)
    } else {
      let a = one_minus_cos! / theta
      let b = 1.0 - sin_theta! / theta
      /// (KK) ^T = K^T  K^T = (-K) (-K) = K*K
      return delta + a * (K! * delta) + b * (KK! * delta)
    }
  }

}

public struct Matrix3Coordinate: ManifoldCoordinate, Equatable, KeyPathIterable {
  public mutating func move(along direction: Vector3) {
  }
  
  /// Exponential map from R3 to SO(3)
  @differentiable(wrt: omega)
  public static func Expmap(_ omega: Vector3) -> Matrix3Coordinate {
    Matrix3Coordinate(ExpmapFunctor(omega).expmap())
  }
  
  /// derivative of expmap
  @derivative(of: Matrix3Coordinate.Expmap)
  @usableFromInline
  static func vjpExpmap(_ omega: Vector3)
    -> (value: Matrix3Coordinate, pullback: (Matrix3Coordinate.TangentVector) -> Vector3)
  {
    let functor = ExpmapFunctor(omega)
    return (value:Matrix3Coordinate(functor.expmap()), pullback:functor.pullbackDexp)
  }

  /// Compose with the exponential map
  @differentiable(wrt: local)
  public func global(_ local: Vector3) -> Matrix3Coordinate {
    Matrix3Coordinate(matmul(self.R, Matrix3Coordinate.Expmap(local).R))
  }
  
  @differentiable(wrt: global)
  public func local(_ global: Matrix3Coordinate) -> Vector3 {
    Vector3(1,2,3)
  }
  
  public typealias LocalCoordinate = Vector3
  public typealias TangentVector = Vector3

  public var R : Tensor<Double>
  
  /// Default constructor creates identity
  public init() {
    R = eye(rowCount:3)
  }
  
  /// Construct from Tensor
  public init(_ tensor: Tensor<Double>) {
    R = tensor
  }
  
  /// Construct from a rotation matrix, as doubles in *row-major* order
  public init(_ r11 : Double, _ r12 : Double, _ r13 : Double,
              _ r21 : Double, _ r22 : Double, _ r23 : Double,
              _ r31 : Double, _ r32 : Double, _ r33 : Double) {
    R = Tensor<Double>(shape: [3,3], scalars: [r11,r12,r13, r21,r22,r23, r31,r32,r33])
  }
}

/// Rot3 class is the Swift type for the SO(3) manifold of 3D rotations.
public struct Rot3 : Manifold, Equatable, KeyPathIterable {
  /// Move along the given direction (retract)
  public mutating func move(along direction: Vector3) {
    coordinateStorage = coordinateStorage.global(direction)
  }
  
  /// SO(3) is a 3-dimensional manifold, tangent vector type = Vector3
  public typealias TangentVector = Vector3
  
  public var coordinateStorage : Matrix3Coordinate
  
  /// Default constructor creates identity
  public init() {
    self.coordinateStorage = Matrix3Coordinate()
  }
  
  /// Default constructor from coordinate storage
  public init(coordinateStorage coordinate: Matrix3Coordinate) {
    self.coordinateStorage = coordinate
  }
  
  /// Construct from a rotation matrix, as doubles in *row-major* order
  public init(_ r11 : Double, _ r12 : Double, _ r13 : Double,
              _ r21 : Double, _ r22 : Double, _ r23 : Double,
              _ r31 : Double, _ r32 : Double, _ r33 : Double) {
    self.coordinateStorage = Matrix3Coordinate(r11,r12,r13, r21,r22,r23, r31,r32,r33)
  }
}

// Boilerplate code for running the test cases
import XCTest

final class Matrix3CoordinateTests: XCTestCase {
  // test exponential map with zero argument
  func testExpmap0() {
    XCTAssertEqual(Matrix3Coordinate.Expmap(Vector3(0,0,0)).R, eye(rowCount: 3))
  }
  // test exponential map with (.pi/2,0,0)
  func testExpmap1() {
    let expected = Tensor<Double>(shape: [3,3], scalars: [1, 0, 0,  0, 0, -1,  0, 1, 0])
    assertEqual(Matrix3Coordinate.Expmap(Vector3(.pi/2,0,0)).R, expected, accuracy: 1e-9)
  }
}

final class Rot3Tests: XCTestCase {
  
  // test constructor from doubles
  func testConstructorEquality() {
    let R1 = Rot3()
    let R2 = Rot3(1, 0, 0,  0, 1, 0,  0, 0, 1)
    XCTAssertEqual(R1, R2)
  }
  
  // test constructor from doubles, non-identity
  func testConstructorInequal() {
    let R1 = Rot3()
    let R2 = Rot3(0, 1, 0, 1, 0, 0, 0, 0, -1)
    XCTAssertNotEqual(R1, R2)
  }
  
  // test that move really works
  func testMove() {
    let xi = Vector3(.pi,0,0)
    var actual = Rot3()
    actual.move(along: xi)
    let expected = Rot3(1, 0, 0,  0, 0, 1,  0, 1, 0)
    XCTAssertEqual(actual, expected)
  }
  
  static var allTests = [
    ("testConstructorEquality", testConstructorEquality),
    ("testConstructorInequal", testConstructorInequal),
  ]
}
