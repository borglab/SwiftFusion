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
  mutating func infer() -> [OrientedBoundingBox] {
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
