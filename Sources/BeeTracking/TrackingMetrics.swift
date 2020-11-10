import SwiftFusion

/// Measurements of how good a predicted track is relative to the ground truth.
public struct TrackingMetrics {
  /// The distances between each frame's predicted and ground truth centers, for the prefix of
  /// frames where the prediction is within a threshold of the ground truth.
  public var poseErrors: [Double] = []

  /// The average of `poseErrors`.
  public var averagePoseError: Double

  /// The first frame where the distance between the predicted and ground truth centers exceeds a
  /// threshold.
  public var trackingFailureFrame: Int? = nil

  /// Creates an instance with the given `groundTruth` and `prediction`.
  ///
  /// Parameter failureThreshold: the distance threshold for a tracking failure.
  public init(
    groundTruth: [OrientedBoundingBox],
    prediction: [OrientedBoundingBox],
    failureThreshold: Double = 10
  ) {
    for (frame, (gt, p)) in zip(groundTruth, prediction).enumerated() {
      let poseError = gt.center.localCoordinate(p.center).norm
      if poseError > failureThreshold {
        trackingFailureFrame = frame
        break
      }
      poseErrors.append(poseError)
    }
    averagePoseError = poseErrors.reduce(0, +) / Double(poseErrors.count)
  }
}
