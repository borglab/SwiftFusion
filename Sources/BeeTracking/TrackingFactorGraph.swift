import BeeDataset
import PenguinStructures
import SwiftFusion
import TensorFlow

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

/// A specification for a factor graph that tracks a target in a sequence of frames.
public struct TrackingConfiguration<FrameVariables: VariableTuple> {
  /// The frames of the video to track.
  public let frames: [Tensor<Float>]

  /// A collection of arbitrary values for the variables in the factor graph.
  public let variableTemplate: VariableAssignments

  /// The ids of the variables in each frame.
  public let frameVariableIDs: [FrameVariables.Indices]

  /// Adds to `graph` a prior factor on `variables`
  public let addPriorFactor: (
    _ variables: FrameVariables.Indices, _ values: FrameVariables, _ graph: inout FactorGraph
  ) -> ()

  /// Adds to `graph` a tracking factor on `variables` for tracking in `frame`.
  public let addTrackingFactor: (
    _ variables: FrameVariables.Indices, _ frame: Tensor<Float>, _ graph: inout FactorGraph
  ) -> ()

  /// Adds to `graph` between factor(s) between the variables at `variables1` and the variables at `variables2`.
  public let addBetweenFactor: (
    _ variables1: FrameVariables.Indices, _ variables2: FrameVariables.Indices,
    _ graph: inout FactorGraph
  ) -> ()

  /// The optimizer to use during inference.
  public var optimizer = LM()

  /// Creates an instance.
  ///
  /// See the field doc comments for argument docs.
  public init(
    frames: [Tensor<Float>],
    variableTemplate: VariableAssignments,
    frameVariableIDs: [FrameVariables.Indices],
    addPriorFactor: @escaping (
      _ variables: FrameVariables.Indices, _ values: FrameVariables, _ graph: inout FactorGraph
    ) -> (),
    addTrackingFactor: @escaping (
      _ variables: FrameVariables.Indices, _ frame: Tensor<Float>, _ graph: inout FactorGraph
    ) -> (),
    addBetweenFactor: @escaping (
      _ variables1: FrameVariables.Indices, _ variables2: FrameVariables.Indices,
      _ graph: inout FactorGraph
    ) -> ()
  ) {
    self.frames = frames
    self.variableTemplate = variableTemplate
    self.frameVariableIDs = frameVariableIDs
    self.addPriorFactor = addPriorFactor
    self.addTrackingFactor = addTrackingFactor
    self.addBetweenFactor = addBetweenFactor

    self.optimizer.precision = 1e-1
    self.optimizer.max_iteration = 100
    self.optimizer.cgls_precision = 1e-5
  }

  /// Returns a `FactorGraph` for the tracking problem on the frames at `frameIndices`.
  public func graph(on frameIndices: Range<Int>) -> FactorGraph {
    var result = FactorGraph()
    for i in frameIndices {
      addTrackingFactor(frameVariableIDs[i], frames[i], &result)
    }
    for i in frameIndices.dropLast() {
      addBetweenFactor(frameVariableIDs[i], frameVariableIDs[i + 1], &result)
    }
    return result
  }

  /// Returns a prediction.
  public mutating func infer(
    knownStart: FrameVariables
  ) -> VariableAssignments {
    // Set the first variable to the known starting position.
    var x = variableTemplate
    x[frameVariableIDs[0]] = knownStart

    // Initialize the variables one frame at a time. Each iteration intializes the `i+1`-th
    // variable.
    for i in 0..<(frames.count - 1) {
      if i % 10 == 0 {
        print("Inferring for frame \(i + 1) of \(frames.count - 1)")
      }

      // Set the initial guess of the `i+1`-th variable to the value of the previous variable.
      x[frameVariableIDs[i + 1]] = x[frameVariableIDs[i], as: FrameVariables.self]

      // Create a tracking factor graph on just the `i`-tha dn `i+1`-th variables
      var g = graph(on: i..<(i + 2))

      // The `i`-th variable is already initialized well, so add a prior factor that it stays
      // near its current position.
      addPriorFactor(frameVariableIDs[i], x[frameVariableIDs[i]], &g)

      // Optimize the factor graph.
      try? optimizer.optimize(graph: g, initial: &x)
    }

    // We could also do a final optimization on all the variables jointly here.

    return x
  }
}

/// Returns a tracking configuration for a tracker using an RAE.
///
/// Parameter model: The RAE model to use.
/// Parameter statistics: Normalization statistics for the frames.
/// Parameter frames: The frames of the video where we want to run tracking.
/// Parameter targetSize: The size of the target in the frames.
public func makeRAETracker(
  model: DenseRAE,
  statistics: FrameStatistics,
  frames: [Tensor<Float>],
  targetSize: (Int, Int)
) -> TrackingConfiguration<Tuple2<Pose2, Vector10>> {
  var variableTemplate = VariableAssignments()
  var frameVariableIDs = [Tuple2<TypedID<Pose2>, TypedID<Vector10>>]()
  for _ in 0..<frames.count {
    frameVariableIDs.append(
      Tuple2(
        variableTemplate.store(Pose2()),
        variableTemplate.store(Vector10())))
  }
  return TrackingConfiguration(
    frames: frames,
    variableTemplate: variableTemplate,
    frameVariableIDs: frameVariableIDs,
    addPriorFactor: { (variables, values, graph) -> () in
      let (poseID, latentID) = unpack(variables)
      let (pose, latent) = unpack(values)
      graph.store(WeightedPriorFactor(poseID, pose, weight: 1e-2))
      graph.store(WeightedPriorFactor(latentID, latent, weight: 1e2))
    },
    addTrackingFactor: { (variables, frame, graph) -> () in
      let (poseID, latentID) = unpack(variables)
      graph.store(
        AppearanceTrackingFactor<Vector10>(
          poseID, latentID,
          measurement: statistics.normalized(frame),
          appearanceModel: { x in
            model.decode(x.expandingShape(at: 0)).squeezingShape(at: 0)
          },
          appearanceModelJacobian: { x in
            model.decodeJacobian(x.expandingShape(at: 0))
              .reshaped(to: [model.imageHeight, model.imageWidth, model.imageChannels, model.latentDimension])
          },
          targetSize: targetSize))
    },
    addBetweenFactor: { (variables1, variables2, graph) -> () in
      let (poseID1, latentID1) = unpack(variables1)
      let (poseID2, latentID2) = unpack(variables2)
      graph.store(WeightedBetweenFactor(poseID1, poseID2, Pose2(), weight: 1e-2))
      graph.store(WeightedBetweenFactor(latentID1, latentID2, Vector10(), weight: 1e2))
    })
}

/// Returns a tracking configuration for a tracker using an PPCA.
///
/// Parameter model: The PPCA model to use.
/// Parameter statistics: Normalization statistics for the frames.
/// Parameter frames: The frames of the video where we want to run tracking.
/// Parameter targetSize: The size of the target in the frames.
public func makePPCATracker(
  model: PPCA,
  statistics: FrameStatistics,
  frames: [Tensor<Float>],
  targetSize: (Int, Int)
) -> TrackingConfiguration<Tuple2<Pose2, Vector10>> {
  var variableTemplate = VariableAssignments()
  var frameVariableIDs = [Tuple2<TypedID<Pose2>, TypedID<Vector10>>]()
  for _ in 0..<frames.count {
    frameVariableIDs.append(
      Tuple2(
        variableTemplate.store(Pose2()),
        variableTemplate.store(Vector10())))
  }
  return TrackingConfiguration(
    frames: frames,
    variableTemplate: variableTemplate,
    frameVariableIDs: frameVariableIDs,
    addPriorFactor: { (variables, values, graph) -> () in
      let (poseID, latentID) = unpack(variables)
      let (pose, latent) = unpack(values)
      graph.store(WeightedPriorFactor(poseID, pose, weight: 1e-2))
      graph.store(WeightedPriorFactor(latentID, latent, weight: 1e2))
    },
    addTrackingFactor: { (variables, frame, graph) -> () in
      let (poseID, latentID) = unpack(variables)
      graph.store(
        AppearanceTrackingFactor<Vector10>(
          poseID, latentID,
          measurement: statistics.normalized(frame),
          appearanceModel: { x in
            model.decode(x)
          },
          appearanceModelJacobian: { x in
            model.W // .reshaped(to: [targetSize.0, targetSize.1, frames[0].shape[3], model.latent_size])
          },
          targetSize: targetSize
        )
      )
    },
    addBetweenFactor: { (variables1, variables2, graph) -> () in
      let (poseID1, latentID1) = unpack(variables1)
      let (poseID2, latentID2) = unpack(variables2)
      graph.store(WeightedBetweenFactor(poseID1, poseID2, Pose2(), weight: 1e-2))
      graph.store(WeightedBetweenFactor(latentID1, latentID2, Vector10(), weight: 1e2))
    })
}

/// Returns a tracking configuration for a raw pixel tracker.
///
/// Parameter frames: The frames of the video where we want to run tracking.
/// Parameter target: The pixels of the target.
public func makeRawPixelTracker(
  frames: [Tensor<Float>],
  target: Tensor<Float>
) -> TrackingConfiguration<Tuple1<Pose2>> {
  var variableTemplate = VariableAssignments()
  var frameVariableIDs = [Tuple1<TypedID<Pose2>>]()
  for _ in 0..<frames.count {
    frameVariableIDs.append(
      Tuple1(
        variableTemplate.store(Pose2())))
  }
  return TrackingConfiguration(
    frames: frames,
    variableTemplate: variableTemplate,
    frameVariableIDs: frameVariableIDs,
    addPriorFactor: { (variables, values, graph) -> () in
      let poseID = variables.head
      let pose = values.head
      graph.store(WeightedPriorFactor(poseID, pose, weight: 1e0))
    },
    addTrackingFactor: { (variables, frame, graph) -> () in
      let poseID = variables.head
      graph.store(
        RawPixelTrackingFactor(poseID, measurement: frame, target: Tensor<Double>(target)))
    },
    addBetweenFactor: { (variables1, variables2, graph) -> () in
      let poseID1 = variables1.head
      let poseID2 = variables2.head
      graph.store(WeightedBetweenFactor(poseID1, poseID2, Pose2(), weight: 1e0))
    })
}

/// Returns `t` as a Swift tuple.
fileprivate func unpack<A, B>(_ t: Tuple2<A, B>) -> (A, B) {
  return (t.head, t.tail.head)
}
