import _Differentiation

/// The five-parameter camera calibration.
public struct Cal3_S2: Manifold, Equatable {
  public typealias Coordinate = Cal3_S2Coordinate
  public typealias TangentVector = Vector5

  public var coordinateStorage: Cal3_S2Coordinate

  public init() {
    self.init(coordinateStorage: Cal3_S2Coordinate())
  }

  public init(coordinateStorage: Cal3_S2Coordinate) {
    self.coordinateStorage = coordinateStorage
  }

  /// Initializes from individual values.
  public init(fx: Double, fy: Double, s: Double, u0: Double, v0: Double) {
    self.init(coordinateStorage: Cal3_S2Coordinate(fx: fx, fy: fy, s: s, u0: u0, v0: v0))
  }

  /// Moves the parameter values by the specified direction.
  public mutating func move(along direction: Vector5) {
    coordinateStorage = coordinateStorage.retract(direction)
  }
}

/// CameraCalibration conformance.
extension Cal3_S2: CameraCalibration {
  @differentiable
  public func uncalibrate(_ np: Vector2) -> Vector2 {
    coordinate.uncalibrate(np)
  }

  @differentiable
  public func calibrate(_ ip: Vector2) -> Vector2 {
    coordinate.calibrate(ip)
  }
}

extension Cal3_S2 {
  public func zeroTangentVector() -> Cal3_S2.TangentVector {
    return Vector5(0.0, 0.0, 0.0, 0.0, 0.0)
  }
}

/// Manifold coordinate for Cal3_S2.
public struct Cal3_S2Coordinate: Equatable {
  /// Focal length in X direction.
  public var fx: Double

  /// Focal length in Y direction.
  public var fy: Double

  /// Skew factor.
  public var s: Double

  /// Image center in X direction.
  public var u0: Double

  /// Image center in Y direction.
  public var v0: Double

  /// Initializes from individual values.
  public init(fx: Double, fy: Double, s: Double, u0: Double, v0: Double) {
    self.fx = fx
    self.fy = fy
    self.s = s
    self.u0 = u0
    self.v0 = v0
  }

  /// Initializes from a vector.
  public init(_ params: Vector5) {
    self.fx = params.s0
    self.fy = params.s1
    self.s = params.s2
    self.u0 = params.s3
    self.v0 = params.s4
  }

  /// Initializes with default values, corresponding to the identity element.
  public init() {
    self.init(fx: 1.0, fy: 1.0, s: 0.0, u0: 0.0, v0: 0.0)
  }

  /// Returns the parameters as a vector.
  public func asVector() -> Vector5 {
    Vector5(fx, fy, s, u0, v0)
  }
}

/// ManifoldCoordinate conformance.
extension Cal3_S2Coordinate: ManifoldCoordinate {
  public typealias LocalCoordinate = Vector5

  @differentiable(wrt: local)
  public func retract(_ local: Vector5) -> Cal3_S2Coordinate {
    Cal3_S2Coordinate(asVector() + local)
  }

  @differentiable(wrt: global)
  public func localCoordinate(_ global: Cal3_S2Coordinate) -> Vector5 {
    global.asVector() - asVector()
  }
}

/// Operations on a point.
extension Cal3_S2Coordinate {
  @differentiable
  public func uncalibrate(_ np: Vector2) -> Vector2 {
    Vector2(fx * np.x + s * np.y + u0, fy * np.y + v0)
  }

  @differentiable
  public func calibrate(_ ip: Vector2) -> Vector2 {
    let (du, dv) = (ip.x - u0, ip.y - v0)
    let (fxInv, fyInv) = (1.0 / fx, 1.0 / fy)
    return Vector2(fxInv * du - s * fxInv * fyInv * dv, fyInv * dv)
  }
}
