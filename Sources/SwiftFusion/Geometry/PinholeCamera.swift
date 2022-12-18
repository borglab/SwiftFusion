import _Differentiation

/// Pinhole camera model.
public struct PinholeCamera<Calibration: CameraCalibration>: Differentiable {
  /// Camera pose in the world/reference frame.
  public var wTc: Pose3

  /// Camera calibration.
  public var calibration: Calibration

  /// Initializes from camera pose and calibration.
  @differentiable(reverse)
  public init(_ calibration: Calibration, _ wTc: Pose3) {
    self.calibration = calibration
    self.wTc = wTc
  }

  /// Initializes with identity pose.
  @differentiable(reverse)
  public init(_ calibration: Calibration) {
    self.init(calibration, Pose3())
  }

  /// Initializes to default.
  public init() {
    self.init(Calibration(), Pose3())
  }
}

/// Project and backproject.
extension PinholeCamera {
    /// Projects a 3D point in the world frame to 2D point in the image.
  @differentiable(reverse)
  public func project(_ wp: Vector3) -> Vector2 {
    let np: Vector2 = projectToNormalized(wp)
    let ip = calibration.uncalibrate(np)
    return ip
  }

  /// Backprojects a 2D image point into 3D point in the world frame at given depth.
  @differentiable(reverse)
  public func backproject(_ ip: Vector2, _ depth: Double) -> Vector3 {
    let np = calibration.calibrate(ip)
    let cp = Vector3(np.x * depth, np.y * depth, depth)
    let wp = wTc * cp
    return wp
  }

  /// Projects a 3D point in the world frame to 2D normalized coordinate.
  @differentiable(reverse)
  func projectToNormalized(_ wp: Vector3) -> Vector2 {
    projectToNormalized(wp).ip
  }

  /// Computes the derivative of the projection function wrt to self and the point wp.
  @usableFromInline
  @derivative(of: projectToNormalized)
  func vjpProjectToNormalized(_ wp: Vector3) -> 
    (value: Vector2, pullback: (Vector2) -> (TangentVector, Vector3))
  {
    let (ip, cRw, zInv) = projectToNormalized(wp)
    let (u, v) = (ip.x, ip.y)
    let R = cRw.coordinate.R
    return (
      value: ip,
      pullback: { p in 
        let dpose = Vector6(
          p.x * (u * v) + p.y * (1 + v * v),
          p.x * -(1 + u * u) + p.y * -(u * v),
          p.x * v + p.y * -u,
          p.x * -zInv,
          p.y * -zInv,
          p.x * (zInv * u) + p.y * (zInv * v))

        let dpoint = zInv * Vector3(
          p.x * (R[0, 0] - u * R[2, 0]) + p.y * (R[1, 0] - v * R[2, 0]),
          p.x * (R[0, 1] - u * R[2, 1]) + p.y * (R[1, 1] - v * R[2, 1]),
          p.x * (R[0, 2] - u * R[2, 2]) + p.y * (R[1, 2] - v * R[2, 2]))

        return (
          TangentVector(wTc: dpose, calibration: calibration.zeroTangentVector()),
          dpoint)
      }
    )
  }

  /// Projects a 3D point in the world frame to 2D normalized coordinate and returns intermediate values.
  func projectToNormalized(_ wp: Vector3) ->
    (ip: Vector2, cRw: Rot3, zInv: Double)
  {
    // Transform the point to camera coordinate
    let cTw = wTc.inverse()
    let cp = cTw * wp

    // TODO: check for cheirality (whether the point is behind the camera)

    // Project to normalized coordinate
    let zInv = 1.0 / cp.z
    
    return (
      ip: Vector2(cp.x * zInv, cp.y * zInv),
      cRw: cTw.rot,
      zInv: zInv)
  }
}
