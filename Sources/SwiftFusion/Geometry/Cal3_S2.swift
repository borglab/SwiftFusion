
/// The five-parameter camera calibration.
public struct Cal3_S2: CameraCalibration, Equatable {
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

  /// Initializes with default values, corresponding to the identity element.
  public init() {
    self.init(fx: 1.0, fy: 1.0, s: 0.0, u0: 0.0, v0: 0.0)
  }
}

/// CameraCalibration protocol conformance.
extension Cal3_S2 {
  @differentiable
  public func uncalibrate(_ pNormalized: Vector2) -> Vector2 {
    Vector2(
      fx * pNormalized.x + s * pNormalized.y + u0,
      fy * pNormalized.y + v0)
  }

  @differentiable
  public func calibrate(_ pImage: Vector2) -> Vector2 {
    let (du, dv) = (pImage.x - u0, pImage.y - v0)
    let (fxInv, fyInv) = (1.0 / fx, 1.0 / fy)
    return Vector2(
      fxInv * du - s * fxInv * fyInv * dv,
      fyInv * dv)
  }
}
