
/// A protocol for camera calibration parameters.
public protocol CameraCalibration: Differentiable {
  /// Initializes to default (usually identity).
  init()

  /// Converts from image coordinate to normalized coordinate.
  @differentiable
  func calibrate(_ ip: Vector2) -> Vector2

  /// Converts from normalized coordinate to image coordinate.
  @differentiable
  func uncalibrate(_ np: Vector2) -> Vector2
}
