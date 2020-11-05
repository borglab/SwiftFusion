import BeeDataset
import SwiftFusion
import TensorFlow

public struct InferenceConfiguration<G: TrackingFactorGraph> {
  public var video: VOTVideo

  public var optimizer: LM

  public typealias MakeFrameVariables =
    (_ frame: Tensor<Double>, _ region: OrientedBoundingBox) -> G.FrameVariables

  public typealias MakeTrackingFactorGraph =
    (_ frames: [Tensor<Double>], _ initialization: G.FrameVariables, _ guesses: [G.FrameVariables]) -> G

  public var makeFrameVariables: MakeFrameVariables
  public var makeTrackingFactorGraph: MakeTrackingFactorGraph

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
  public mutating func inferAllAtOnce() -> [OrientedBoundingBox] {
    let initialBox = video.track[0]
    let initialization = makeFrameVariables(video.frames[0], initialBox)
    let tracker = makeTrackingFactorGraph(
      video.frames,
      initialization,
      Array(repeating: initialization, count: video.frames.count))
    var v = tracker.v
    try? optimizer.optimize(graph: tracker.fg, initial: &v)
    return tracker.poseIds.map {
      OrientedBoundingBox(center: v[$0], rows: initialBox.rows, cols: initialBox.cols)
    }
  }

  public mutating func inferOneFrameAtATime() -> [OrientedBoundingBox] {
    var predictions = [video.track[0]]
    var frameVariables = [makeFrameVariables(video.frames[0], video.track[0])]

    for i in 0..<(video.frames.count - 1) {
      let videoSlice = video[i..<(i + 2)]
      let tracker = makeTrackingFactorGraph(
        videoSlice.frames,
        frameVariables[i],
        Array(repeating: frameVariables[i], count: 2))
      var v = tracker.v
      try? optimizer.optimize(graph: tracker.fg, initial: &v)
      let prediction = OrientedBoundingBox(
        center: v[tracker.poseIds[1]],
        rows: predictions[0].rows,
        cols: predictions[0].cols)
      predictions.append(prediction)
      frameVariables.append(makeFrameVariables(videoSlice.frames[1], prediction))
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
