import BeeDataset
import SwiftFusion
import TensorFlow

/// Settings for infering a track on a video.
public struct InferenceConfiguration<G: TrackingFactorGraph> {
  /// The video to track.
  public var video: VOTVideo

  /// The optimizer to use during inference.
  public var optimizer: LM

  /// Given a `frame` and a `region` in the frame, returns the factor graph variables (e.g. pose
  /// and latent code) corresponding to that frame.
  public typealias MakeFrameVariables =
    (_ frame: Tensor<Double>, _ region: OrientedBoundingBox) -> G.FrameVariables

  /// Creates a factor graph that tracks the object in `frames` that starts out at `start`
  /// in `frames[0]`.
  ///
  /// Parameter guesses: Initial guesses for the variables in all the frames.
  public typealias MakeTrackingFactorGraph =
    (_ frames: [Tensor<Double>], _ start: G.FrameVariables, _ guesses: [G.FrameVariables]) -> G

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
  /// Initializes a guess that the target is at the starting point in frame 1. Uses LM to optimize
  /// this guess. Takes the solution as a guess for the position in frame 2. Repeats for all
  /// frames.
  public mutating func inferOneFrameAtATime() -> [OrientedBoundingBox] {
    var predictions = [video.track[0]]
    var frameVariables = [makeFrameVariables(video.frames[0], video.track[0])]

    for i in 1..<video.frames.count {
      let videoSlice = video[i..<(i+1)]
      let tracker = makeTrackingFactorGraph(
        videoSlice.frames,
        frameVariables[i - 1],
        Array(repeating: frameVariables[i - 1], count: 1))
      var v = tracker.v
      try? optimizer.optimize(graph: tracker.fg, initial: &v)
      let prediction = OrientedBoundingBox(
        center: v[tracker.poseIds[0]],
        rows: predictions[0].rows,
        cols: predictions[0].cols)
      predictions.append(prediction)
      frameVariables.append(makeFrameVariables(videoSlice.frames[0], prediction))
    }

    return predictions
  }
}

public protocol TrackingFactorGraph {
  associatedtype FrameVariables

  var fg: FactorGraph { get }
  var v: VariableAssignments { get }
  var poseIds: [TypedID<Pose2>] { get }
}

extension AppearanceTrackingFactorGraph: TrackingFactorGraph {
  public typealias FrameVariables = (Pose2, Vector10)
}

extension RawPixelTrackingFactorGraph: TrackingFactorGraph {
  public typealias FrameVariables = Pose2
}
