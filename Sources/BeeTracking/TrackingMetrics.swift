import SwiftFusion

public struct TrackingMetrics {
  public var poseErrors: [Double] = []
  public var averagePoseError: Double
  public var trackingFailureFrame: Int? = nil

  public init(
    groundTruth: [OrientedBoundingBox],
    prediction: [OrientedBoundingBox],
    failureThreshold: Double = 50
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
