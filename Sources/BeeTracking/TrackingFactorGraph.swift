import BeeDataset
import PenguinStructures
import SwiftFusion
import TensorFlow
import PythonKit
import Foundation

public struct WeightedBetweenFactorPose2: LinearizableFactor2 {
  public typealias Pose = Pose2
  public let edges: Variables.Indices
  public let difference: Pose
  public let weight: Double
  public let rotWeight: Double

  public init(_ startId: TypedID<Pose>, _ endId: TypedID<Pose>, _ difference: Pose, weight: Double, rotWeight: Double = 1.0) {
    self.edges = Tuple2(startId, endId)
    self.difference = difference
    self.weight = weight
    self.rotWeight = rotWeight
  }

  @differentiable
  public func errorVector(_ start: Pose, _ end: Pose) -> Pose.TangentVector {
    let actualMotion = between(start, end)
    let weighted = weight * difference.localCoordinate(actualMotion)
    return Vector3(rotWeight * weighted.x, weighted.y, weighted.z)
  }
}

public struct WeightedBetweenFactorPose2SD: LinearizableFactor2 {
  public typealias Pose = Pose2
  public let edges: Variables.Indices
  public let difference: Pose

  public let sdX: Double
  public let sdY: Double
  public let sdTheta: Double

  public init(_ startId: TypedID<Pose>, _ endId: TypedID<Pose>, _ difference: Pose, sdX: Double, sdY: Double, sdTheta: Double) {
    self.edges = Tuple2(startId, endId)
    self.difference = difference
    self.sdX = sdX
    self.sdY = sdY
    self.sdTheta = sdTheta
  }

  @differentiable
  public func errorVector(_ start: Pose, _ end: Pose) -> Pose.TangentVector {
    let actualMotion = between(start, end)
    let local = difference.localCoordinate(actualMotion)
    return Vector3(local.x / sdTheta, local.y / sdX, local.z / sdY)
  }
}

public struct WeightedPriorFactorPose2: LinearizableFactor1 {
  public typealias Pose = Pose2
  public let edges: Variables.Indices
  public let prior: Pose
  public let weight: Double
  public let rotWeight: Double

  public init(_ startId: TypedID<Pose>, _ prior: Pose, weight: Double, rotWeight: Double = 1.0) {
    self.edges = Tuple1(startId)
    self.prior = prior
    self.weight = weight
    self.rotWeight = rotWeight
  }

  @differentiable
  public func errorVector(_ start: Pose) -> Pose.TangentVector {
    let weighted = weight * prior.localCoordinate(start)
    return Vector3(rotWeight * weighted.x, weighted.y, weighted.z)
  }
}

public struct WeightedPriorFactorPose2SD: LinearizableFactor1 {
  public typealias Pose = Pose2
  public let edges: Variables.Indices
  public let prior: Pose
  public let sdX: Double
  public let sdY: Double
  public let sdTheta: Double

  public init(_ startId: TypedID<Pose>, _ prior: Pose, sdX: Double, sdY: Double, sdTheta: Double) {
    self.edges = Tuple1(startId)
    self.prior = prior
    self.sdX = sdX
    self.sdY = sdY
    self.sdTheta = sdTheta
  }

  @differentiable
  public func errorVector(_ start: Pose) -> Pose.TangentVector {
    let local = prior.localCoordinate(start)
    return Vector3(local.x / sdTheta, local.y / sdX, local.z / sdY)
  }
}

/// A specification for a factor graph that tracks a target in a sequence of frames.
public struct TrackingConfiguration<FrameVariables: VariableTuple> {
  /// The frames of the video to track.
  public var frames: [Tensor<Float>]

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

  /// Adds to `graph` "between factor(s)" between `constantVariables` and `variables` that treat
  /// the `constantVariables` as fixed.
  ///
  /// This is used during frame-by-frame initialization to constrain frame `i + 1` by a between
  /// factor on the value from frame `i` without optimizing the value of frame `i`.
  public let addFixedBetweenFactor: (
    _ values: FrameVariables, _ variables: FrameVariables.Indices,
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
    ) -> (),
    addFixedBetweenFactor: ((
      _ values: FrameVariables, _ variables: FrameVariables.Indices,
      _ graph: inout FactorGraph
    ) -> ())? = nil
  ) {
    precondition(
      addFixedBetweenFactor != nil,
      "I added a runtime check for this argument so that I would not have to change all " +
        "callers before compiling. It is actually required."
    )

    self.frames = frames
    self.variableTemplate = variableTemplate
    self.frameVariableIDs = frameVariableIDs
    self.addPriorFactor = addPriorFactor
    self.addTrackingFactor = addTrackingFactor
    self.addBetweenFactor = addBetweenFactor
    self.addFixedBetweenFactor = addFixedBetweenFactor!

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
  
  // Try to initialize pose of the `i+1`-th variable by sampling
  mutating func extendBySampling(x: inout VariableAssignments, fromFrame i:Int, withGraph g: FactorGraph, numberOfSamples: Int = 2000, perturbVector: Vector3 = Vector3(0.3, 8, 4.6))  {
    // First get pose IDs: pose is assumed to be first variable in the frameVariableID tuple
    let currentPoseID = (frameVariableIDs[i + 1] as! Tuple1<TypedID<Pose2>>).head
    let previousPoseID = (frameVariableIDs[i] as! Tuple1<TypedID<Pose2>>).head
    
    // Remember best pose
    var bestPose = x[currentPoseID]
    
    // Sample from motion model and take best pose
    var bestError = g.error(at: x)
    for _ in 0..<numberOfSamples {
      x[currentPoseID] = x[previousPoseID]
      x[currentPoseID].perturbWith(stddev: perturbVector)
      let candidateError = g.error(at: x)
      if candidateError < bestError {
        bestError = candidateError
        bestPose = x[currentPoseID]
      }
    }
    x[currentPoseID] = bestPose
  }
  
  /// Extend the track
  mutating func extendTrack(x: inout VariableAssignments, fromFrame i:Int,
                            withSampling samplingFlag: Bool = false
  ) {
    let currentVarID = frameVariableIDs[i + 1]
    let previousVarID = frameVariableIDs[i]
    
    // Create a tracking factor graph on just the `i+1`-th variable.
    var g = graph(on: (i + 1)..<(i + 2))
    
    // The `i`-th variable is already initialized well, so add a prior factor that it stays
    // near its current position.
    addFixedBetweenFactor(x[previousVarID], currentVarID, &g)
    
    // Initialize
    if (samplingFlag) {
      // Try to initialize pose of the `i+1`-th variable by sampling
      extendBySampling(x: &x, fromFrame:i, withGraph:g)
    } else {
      // Initialize `i+1`-th variable with the value of the previous variable.
      x[currentVarID] = x[previousVarID, as: FrameVariables.self]
    }
    
    // Optimize the factor graph.
    try? optimizer.optimize(graph: g, initial: &x)
  }
  
  /// Returns a prediction.
  public mutating func infer(
    knownStart: FrameVariables,
    withSampling samplingFlag: Bool = false
  ) -> VariableAssignments {
    // Set the first variable to the known starting position.
    var x = variableTemplate
    x[frameVariableIDs[0]] = knownStart
    
    // Initialize the variables one frame at a time. Each iteration intializes the `i+1`-th
    // variable.
    for i in 0..<(frames.count - 1) {
      print("Inferring for frame \(i + 1) of \(frames.count - 1)")
      extendTrack(x: &x, fromFrame:i, withSampling:samplingFlag)
    }
    
    // TODO: We could also do a final optimization on all the variables jointly here.
    
    return x
  }
}

/// Get the foreground and background batches
public func getTrainingBatches(
  dataset: OISTBeeVideo, boundingBoxSize: (Int, Int),
  fgBatchSize: Int = 3000,
  bgBatchSize: Int = 3000,
  fgRandomFrameCount: Int = 10,
  bgRandomFrameCount: Int = 10,
  useCache: Bool = false
) -> (fg: Tensor<Double>, bg: Tensor<Double>, statistics: FrameStatistics) {
  precondition(dataset.frames.count >= bgRandomFrameCount)
  let np = Python.import("numpy")

  var statistics = FrameStatistics(Tensor<Double>(0.0))
  statistics.mean = Tensor(62.26806976644069)
  statistics.standardDeviation = Tensor(37.44683834503672)

  /// Caching
  let cachePath = "./training_batch_cache"
  let cacheURL = URL(fileURLWithPath: "\(cachePath)_fg.npy")
  if useCache && FileManager.default.fileExists(atPath: cacheURL.path) {
    let foregroundBatch = Tensor<Double>(numpy: np.load("\(cachePath)_fg.npy"))!
    let backgroundBatch = Tensor<Double>(numpy: np.load("\(cachePath)_bg.npy"))!

    precondition(foregroundBatch.shape[0] == fgBatchSize, "Wrong foreground dataset cache, please delete and regenerate!")
    precondition(backgroundBatch.shape[0] == bgBatchSize, "Wrong background dataset cache, please delete and regenerate!")
    
    return (fg: foregroundBatch, bg: backgroundBatch, statistics: statistics)
  }

  let foregroundBatch = dataset.makeBatch(
    statistics: statistics, appearanceModelSize: boundingBoxSize,
    randomFrameCount: fgRandomFrameCount, batchSize: fgBatchSize
  )

  let backgroundBatch = dataset.makeBackgroundBatch(
    patchSize: boundingBoxSize, appearanceModelSize: boundingBoxSize,
    statistics: statistics,
    randomFrameCount: bgRandomFrameCount,
    batchSize: bgBatchSize
  )

  if useCache {
    np.save("\(cachePath)_fg", foregroundBatch.makeNumpyArray())
    np.save("\(cachePath)_bg", backgroundBatch.makeNumpyArray())
  }

  return (fg: foregroundBatch, bg: backgroundBatch, statistics: statistics)
}

/// Train a random projection tracker with a full Gaussian foreground model
/// and a Naive Bayes background model.
/// If EMflag is true we will do several iterations of Monte Carlo EM
public func trainRPTracker(trainingData: OISTBeeVideo,
                           frames: [Tensor<Float>],
                           boundingBoxSize: (Int, Int),
                           withFeatureSize d: Int,
                           fgRandomFrameCount: Int = 10,
                           bgRandomFrameCount: Int = 10,
                           usingEM EMflag: Bool = true
) -> TrackingConfiguration<Tuple1<Pose2>> {
  let (fg, bg, statistics) = getTrainingBatches(
    dataset: trainingData, boundingBoxSize: boundingBoxSize,
    fgRandomFrameCount: fgRandomFrameCount,
    bgRandomFrameCount: bgRandomFrameCount
  )
  
  let randomProjector = RandomProjection(
    fromShape: [boundingBoxSize.0, boundingBoxSize.1, 1], toFeatureSize: d
  )

  let batchPositive = randomProjector.encode(fg)
  let foregroundModel = MultivariateGaussian(from: batchPositive, given: 1e-3)

  let batchNegative = randomProjector.encode(bg)
  let backgroundModel = GaussianNB(from: batchNegative, given: 1e-3)

  let tracker = makeRandomProjectionTracker(
    model: randomProjector, statistics: statistics,
    frames: frames, targetSize: boundingBoxSize,
    foregroundModel: foregroundModel, backgroundModel: backgroundModel
  )
  
  return tracker
}

/// Given a trained tracker, run the tracker on a given number of frames on the test set
public func createSingleTrack(
  onTrack trackId: Int,
  withTracker tracker: inout TrackingConfiguration<Tuple1<Pose2>>,
  andTestData testData: OISTBeeVideo,
  withSampling samplingFlag: Bool = false
) -> ([Pose2], [Pose2]) {
  precondition(trackId < testData.tracks.count, "specified track does not exist!!!")

  let startPose = testData.tracks[trackId].boxes[0].center
  let prediction = tracker.infer(knownStart: Tuple1(startPose), withSampling: samplingFlag)
  let track = tracker.frameVariableIDs.map { prediction[unpack($0)] }
  let groundTruth = testData.tracks[trackId].boxes.map { $0.center }
  return (track, groundTruth)
}

/// Runs the random projections tracker
/// Given a training set, it will train an RP tracker
/// and run it on one track in the test set:
///  - output: image with track and overlap metrics
public func runRPTracker(
  directory: URL, onTrack trackIndex: Int, forFrames: Int = 80,
  withSampling samplingFlag: Bool = false,
  usingEM EMflag: Bool = true,
  withFeatureSize d: Int = 100,
  savePatchesIn resultDirectory: String? = nil
) -> (fig: PythonObject, track: [Pose2], groundTruth: [Pose2]) {
  // train foreground and background model and create tracker
  let trainingData = OISTBeeVideo(directory: directory, length: 100)!
  let testData = OISTBeeVideo(directory: directory, afterIndex: 100, length: forFrames)!
  
  precondition(testData.tracks[trackIndex].boxes.count == forFrames, "track length and required does not match")
  
  var tracker = trainRPTracker(
    trainingData: trainingData,
    frames: testData.frames, boundingBoxSize: (40, 70), withFeatureSize: d, usingEM :EMflag
  )
  
  // Run the tracker and return track with ground truth
  let (track, groundTruth) = createSingleTrack(
    onTrack: trackIndex, withTracker: &tracker,
    andTestData: testData, withSampling: samplingFlag
  )
  
  // Now create trajectory and metrics plot
  let plt = Python.import("matplotlib.pyplot")
  let (fig, axes) = plt.subplots(2, 1, figsize: Python.tuple([6, 12])).tuple2
  plotTrajectory(
    track: track, withGroundTruth: groundTruth, on: axes[0],
    withTrackColors: plt.cm.jet, withGtColors: plt.cm.gray
  )
  
  plotOverlap(
    track: track, withGroundTruth: groundTruth, on: axes[1]
  )

  if let dir = resultDirectory {
    /// Plot all the frames so we can visually inspect the situation
    for i in track.indices {
      let (fig_initial, _) = plotPatchWithGT(frame: testData.frames[i], actual: track[i], expected: groundTruth[i])
      fig_initial.savefig("\(dir)/frank02_1st_img_track\(trackIndex)_\(d)_\(i).png", bbox_inches: "tight")
    }
  }

  return (fig, track, groundTruth)
}

/// Returns `t` as a Swift tuple.
fileprivate func unpack<A, B>(_ t: Tuple2<A, B>) -> (A, B) {
  return (t.head, t.tail.head)
}
/// Returns `t` as a Swift tuple.
fileprivate func unpack<A>(_ t: Tuple1<A>) -> (A) {
  return (t.head)
}
