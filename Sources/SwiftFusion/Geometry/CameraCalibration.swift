
/// A type for camera calibration parameters.
public protocol CameraCalibration: Differentiable {
  /// Initializes to identity.
  init()

  /// Converts from image coordinate to normalized coordinate.
  @differentiable
  func calibrate(_ pImage: Vector2) -> Vector2

  /// Converts from normalized coordinate to image coordinate.
  @differentiable
  func uncalibrate(_ pNormalized: Vector2) -> Vector2
}
