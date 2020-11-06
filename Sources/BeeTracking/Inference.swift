import BeeDataset
import SwiftFusion
import TensorFlow

/// Settings for infering a track on a video.
public struct InferenceConfiguration<G: TrackingFactorGraph, FrameVariables> {
  /// The video to track.
  public var video: VOTVideo

  /// The optimizer to use during inference.
  public var optimizer: LM

  /// Given a `frame` and a `region` in the frame, returns the factor graph variables (e.g. pose
  /// and latent code) corresponding to that frame.
  public typealias MakeFrameVariables =
    (_ frame: Tensor<Double>, _ region: OrientedBoundingBox) -> FrameVariables

  /// Creates a factor graph that tracks the object in `frames` that starts out at `start`
  /// in `frames[0]`.
  ///
  /// Parameter guesses: Initial guesses for the variables in all the frames.
  public typealias MakeTrackingFactorGraph =
    (_ frames: [Tensor<Double>], _ start: FrameVariables, _ guesses: [FrameVariables]) -> G

  /// See `MakeFrameVariables`.
  public var makeFrameVariables: MakeFrameVariables

  /// See `MakeTrackingFactorGraph`.
  public var makeTrackingFactorGraph: MakeTrackingFactorGraph

  /// Creates a instance with the given settings.
  public init(
    video: VOTVideo,
    optimizer: LM,
    makeFrameVariables: @escaping MakeFrameVariables,
    makeTrackingFactorGraph: @escaping MakeTrackingFactorGraph
  ) {
    self.video = video
    self.optimizer = optimizer
    self.makeFrameVariables = makeFrameVariables
    self.makeTrackingFactorGraph = makeTrackingFactorGraph
  }
}

extension InferenceConfiguration {
  /// Returns a prediction.
  ///
  /// Initialzes a guess that the target stays at the starting point throughout the whole video.
  /// Then, jointly optimizes all the variables in the guess using LM.
  public mutating func inferAllAtOnce() -> [OrientedBoundingBox] {
    let initialBox = video.track[0]
    let start = makeFrameVariables(video.frames[0], initialBox)
    let tracker = makeTrackingFactorGraph(
      video.frames,
      start,
      Array(repeating: start, count: video.frames.count))
    var v = tracker.v
    try? optimizer.optimize(graph: tracker.fg, initial: &v)
    return tracker.poseIds.map {
      OrientedBoundingBox(center: v[$0], rows: initialBox.rows, cols: initialBox.cols)
    }
  }

  /// Returns a prediction.
  ///
  /// Slides a window over the frames. For each window, initializes a guess using the predictions
  /// from the previous window and then jointly optimizes all the variables in the window.
  public mutating func inferSlidingWindow(windowSize: Int, sampleSteps: Int = 0) -> [OrientedBoundingBox] {
    var predictions = [video.track[0]]

    for windowEnd in 1..<video.frames.count {
      let windowStart = max(0, windowEnd - windowSize)
      let videoSlice = video[windowStart..<windowEnd]

      // Use the last prediction from the previous window as the initial guess for the position
      // of the target in the new frame that appeared in this window.
      if predictions.count < windowEnd { predictions.append(predictions.last!) }

      // Use the current predictions for the window to initialize the variables in the factor
      // graph.
      let guess = predictions[windowStart..<windowEnd].enumerated().map {
        makeFrameVariables(videoSlice.frames[$0.0], $0.1)
      }

      // Optimize the factor graph.
      let tracker = makeTrackingFactorGraph(videoSlice.frames, guess[0], guess)
      let lastPose = tracker.sampleLastPose(steps: sampleSteps)
      var v = tracker.v
      v[tracker.poseIds.last!] = lastPose
      try? optimizer.optimize(graph: tracker.fg, initial: &v)

      // Replace the predictions with the new solution.
      predictions.replaceSubrange(windowStart..<windowEnd, with: tracker.poseIds.map {
        OrientedBoundingBox(center: v[$0], rows: predictions[0].rows, cols: predictions[0].cols)
      })
    }

    return predictions
  }
}

public protocol TrackingFactorGraph {
  var fg: FactorGraph { get }
  var v: VariableAssignments { get }
  var poseIds: [TypedID<Pose2>] { get }
}

extension TrackingFactorGraph {
  public func sampleLastPose(steps: Int) -> Pose2 {
    var current = v
    var currentError = fg.error(at: current)
    for _ in 0..<steps {
      let offset = Pose2.TangentVector(0, Double.random(in: -50.0..<50.0), Double.random(in: -50.0..<50.0))
      var next = current
      next[poseIds.last!].move(along: offset)
      let nextError = fg.error(at: next)
      let logAcceptanceRate = currentError - nextError
      if logAcceptanceRate > 0 || Double.random(in: 0.0..<1.0) < exp(logAcceptanceRate) {
        current = next
        currentError = nextError
      }
    }
    return current[poseIds.last!]
  }
}

extension AppearanceTrackingFactorGraph: TrackingFactorGraph {}

extension RawPixelTrackingFactorGraph: TrackingFactorGraph {}
