import BeeDataset
import PenguinStructures
import SwiftFusion

/// A factor that specifies a prior on a pose.
public struct WeightedPriorFactor<Pose: LieGroup>: LinearizableFactor1 {
  public let edges: Variables.Indices
  public let prior: Pose
  public let weight: Double

  public init(_ id: TypedID<Pose>, _ prior: Pose, weight: Double) {
    self.edges = Tuple1(id)
    self.prior = prior
    self.weight = weight
  }

  @differentiable
  public func errorVector(_ x: Pose) -> Pose.TangentVector {
    return weight * prior.localCoordinate(x)
  }
}

/// A factor that specifies a difference between two poses.
public struct WeightedBetweenFactor<Pose: LieGroup>: LinearizableFactor2 {
  public let edges: Variables.Indices
  public let difference: Pose
  public let weight: Double

  public init(_ startId: TypedID<Pose>, _ endId: TypedID<Pose>, _ difference: Pose, weight: Double) {
    self.edges = Tuple2(startId, endId)
    self.difference = difference
    self.weight = weight
  }

  @differentiable
  public func errorVector(_ start: Pose, _ end: Pose) -> Pose.TangentVector {
    let actualMotion = between(start, end)
    return weight * difference.localCoordinate(actualMotion)
  }
}

/// A factor graph that tracks a target using an appearance model.
public struct TrackingFactorGraph {
  /// The factors.
  public var fg = FactorGraph()

  /// An initial guess for the variable values.
  public var v = VariableAssignments()

  /// The ids of the poses in the factor graph.
  public var poseIds: [TypedID<Pose2>] = []

  /// The ids of the latent codes in the factor graph.
  public var latentIds: [TypedID<Vector10>] = []

  public struct WeightConfiguration {
    public let latent: Double = 1e3
    public let pose: Double = 1e3
    public init() {}
  }

  /// Create an instance that tracks `trackId` in `video`, starting at `indexStart`, for `length`
  /// steps.
  ///
  /// Parameter model: The appearance model.
  /// Parameter statistics: The statistics used to normalize the frames before encoding.
  public init(
    _ model: DenseRAE,
    _ video: BeeVideo,
    _ statistics: FrameStatistics,
    trackId: Int,
    indexStart: Int,
    length: Int,
    weights: WeightConfiguration = WeightConfiguration()
  ) {
    let expectedLatentDimension = 10
    precondition(
      model.latentDimension == expectedLatentDimension,
      "expected latent dimension \(expectedLatentDimension) but got \(model.latentDimension)")
    precondition(
      trackId < video.tracks.count,
      "track \(trackId) out of bounds, there are \(video.tracks.count) tracks")
    precondition(
      indexStart + length <= video.tracks[trackId].count,
      "final index \(indexStart + length) out of bounds, " +
        "there are \(video.tracks[trackId].count) frames")

    let initialPatch = statistics.normalized(
      video.loadFrame(indexStart)!.patch(at: video.tracks[trackId][indexStart].location))
    let initialLatent = Vector10(flatTensor: model.encode(initialPatch.expandingShape(at: 0)))

    for i in 0..<length {
      poseIds.append(v.store(video.tracks[trackId][indexStart].location.center))
      latentIds.append(v.store(initialLatent))

      fg.store(
        AppearanceTrackingFactor<Vector10>(
          poseIds[i], latentIds[i],
          measurement: statistics.normalized(video.loadFrame(indexStart + i)!),
          appearanceModel: { x in
            model.decode(x.expandingShape(at: 0)).squeezingShape(at: 0)
          },
          appearanceModelJacobian: { x in
            model.decodeJacobian(x.expandingShape(at: 0))
              .reshaped(to: [model.imageHeight, model.imageWidth, model.imageChannels, model.latentDimension])
          }))

      if i == 0 {
        fg.store(WeightedPriorFactor(latentIds[i], initialLatent, weight: weights.latent))
      }
    }

    for i in 0..<(length-1) {
      fg.store(WeightedBetweenFactor<Vector10>(latentIds[i], latentIds[i+1], Vector10.zero, weight: weights.latent))
      fg.store(WeightedBetweenFactor<Pose2>(poseIds[i], poseIds[i+1], Pose2(0,0,0), weight: weights.pose))
    }
  }

  /// Create an instance that tracks `video`, starting at `indexStart`, for `length` steps.
  ///
  /// Parameter model: The appearance model.
  /// Parameter statistics: The statistics used to normalize the frames before encoding.
  public init(
    _ model: DenseRAE,
    _ video: VOTVideo,
    _ statistics: FrameStatistics,
    indexStart: Int,
    length: Int,
    patchSize: (Int, Int),
    weights: WeightConfiguration = WeightConfiguration()
  ) {
    let expectedLatentDimension = 10
    precondition(
      model.latentDimension == expectedLatentDimension,
      "expected latent dimension \(expectedLatentDimension) but got \(model.latentDimension)")
    precondition(
      indexStart + length <= video.track.count,
      "final index \(indexStart + length) out of bounds, " +
        "there are \(video.track.count) frames")

    let initialPatch = statistics.normalized(
      video.frames[indexStart].patch(at: video.track[indexStart], outputSize: patchSize))
    let initialLatent = Vector10(flatTensor: model.encode(initialPatch.expandingShape(at: 0)))

    for i in 0..<length {
      poseIds.append(v.store(video.track[indexStart].center))
      latentIds.append(v.store(initialLatent))

      fg.store(
        AppearanceTrackingFactor<Vector10>(
          poseIds[i], latentIds[i],
          measurement: statistics.normalized(video.frames[indexStart + i]),
          appearanceModel: { x in
            model.decode(x.expandingShape(at: 0)).squeezingShape(at: 0)
          },
          appearanceModelJacobian: { x in
            model.decodeJacobian(x.expandingShape(at: 0))
              .reshaped(to: [model.imageHeight, model.imageWidth, model.imageChannels, model.latentDimension])
          }))

      if i == 0 {
        fg.store(WeightedPriorFactor(latentIds[i], initialLatent, weight: weights.latent))
      }
    }

    for i in 0..<(length-1) {
      fg.store(WeightedBetweenFactor<Vector10>(latentIds[i], latentIds[i+1], Vector10.zero, weight: weights.latent))
      fg.store(WeightedBetweenFactor<Pose2>(poseIds[i], poseIds[i+1], Pose2(0,0,0), weight: weights.pose))
    }
  }
}
