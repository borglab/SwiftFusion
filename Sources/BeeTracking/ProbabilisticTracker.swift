import BeeDataset
import PenguinStructures
import SwiftFusion
import TensorFlow
import PythonKit
import Foundation

// TODO: Move these functions to something like ProbabilistTrackingUtilities.swift

/// Runs the random projections tracker
/// Given a training set, it will train an RP tracker
/// and run it on one track in the test set:
///  - output: image with track and overlap metrics
public func runProbabilisticTracker<
  Encoder: AppearanceModelEncoder,
  ForegroundModel: GenerativeDensity,
  BackgroundModel: GenerativeDensity
>(
  directory: URL,
  likelihoodModel: TrackingLikelihoodModel<Encoder, ForegroundModel, BackgroundModel>,
  onTrack trackIndex: Int,
  forFrames: Int = 80,
  withSampling samplingFlag: Bool = false,
  withFeatureSize d: Int = 100,
  savePatchesIn resultDirectory: String? = nil
) -> (fig: PythonObject, track: [Pose2], groundTruth: [Pose2]) {
  let testSetStart = 100
  
  // Create tracker
  let testData = OISTBeeVideo(directory: directory, afterIndex: testSetStart, length: forFrames)!

  precondition(testData.tracks[trackIndex].boxes.count == forFrames, "track length and required does not match")

  var tracker = makeProbabilisticTracker(
    model: likelihoodModel.encoder,
    frames: testData.frames, targetSize: (40, 70),
    foregroundModel: likelihoodModel.foregroundModel, backgroundModel: likelihoodModel.backgroundModel
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
      fig_initial.savefig("\(dir)/track\(trackIndex)_\(d)_\(i).png", bbox_inches: "tight")
      plt.close(fig: fig_initial)
    }
  }

  return (fig, track, groundTruth)
}

/// Runs the random projections tracker
/// Given a training set, it will train an RP tracker
/// and run it on one track in the test set:
///  - output: image with track and overlap metrics
public func runProbabilisticTracker<Encoder: AppearanceModelEncoder>(
  directory: URL,
  encoder: Encoder,
  onTrack trackIndex: Int,
  forFrames: Int = 80,
  withSampling samplingFlag: Bool = false,
  withFeatureSize d: Int = 100,
  savePatchesIn resultDirectory: String? = nil
) -> (fig: PythonObject, track: [Pose2], groundTruth: [Pose2]) {
  let trainingDatasetSize = 100
  let testSetStart = 100
  
  precondition(trainingDatasetSize <= testSetStart)

  // train foreground and background model and create tracker
  let trainingData = OISTBeeVideo(directory: directory, length: trainingDatasetSize)!
  let testData = OISTBeeVideo(directory: directory, afterIndex: testSetStart, length: forFrames)!

  precondition(testData.tracks[trackIndex].boxes.count == forFrames, "track length and required does not match")
  
  var tracker = trainProbabilisticTracker(
    trainingData: trainingData,
    encoder: encoder,
    frames: testData.frames,
    boundingBoxSize: (40, 70),
    withFeatureSize: d,
    fgRandomFrameCount: trainingDatasetSize,
    bgRandomFrameCount: trainingDatasetSize,
    numberOfTrainingSamples: 3000
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
      fig_initial.savefig("\(dir)/track\(trackIndex)_\(d)_\(i).png", bbox_inches: "tight")
      plt.close(fig: fig_initial)
    }
  }

  return (fig, track, groundTruth)
}

/// Train a random projection tracker with a full Gaussian foreground model
/// and a Naive Bayes background model.
public func trainProbabilisticTracker<Encoder: AppearanceModelEncoder>(
  trainingData: OISTBeeVideo,
  encoder: Encoder,
  frames: [Tensor<Float>],
  boundingBoxSize: (Int, Int), withFeatureSize d: Int,
  fgRandomFrameCount: Int = 10,
  bgRandomFrameCount: Int = 10,
  numberOfTrainingSamples: Int = 3000
) -> TrackingConfiguration<Tuple1<Pose2>> {
  let (fg, bg, _) = getTrainingBatches(
    dataset: trainingData, boundingBoxSize: boundingBoxSize,
    fgBatchSize: numberOfTrainingSamples,
    bgBatchSize: numberOfTrainingSamples,
    fgRandomFrameCount: fgRandomFrameCount,
    bgRandomFrameCount: bgRandomFrameCount,
    useCache: true
  )
  let batchPositive = encoder.encode(fg)
  let foregroundModel = MultivariateGaussian(from:batchPositive, regularizer: 1e-3)

  let batchNegative = encoder.encode(bg)
  let backgroundModel = MultivariateGaussian(from: batchNegative, regularizer: 1e-3)

  let tracker = makeProbabilisticTracker(
    model: encoder,
    frames: frames, targetSize: boundingBoxSize,
    foregroundModel: foregroundModel, backgroundModel: backgroundModel
  )

  return tracker
}



/// Returns a tracking configuration for a tracker using an random projection.
///
/// Parameter model: The random projection model to use.
/// Parameter frames: The frames of the video where we want to run tracking.
/// Parameter targetSize: The size of the target in the frames.
public func makeProbabilisticTracker<
  Encoder: AppearanceModelEncoder,
  ForegroundModel: GenerativeDensity,
  BackgroundModel: GenerativeDensity
>(
  model: Encoder,
  frames: [Tensor<Float>],
  targetSize: (Int, Int),
  foregroundModel: ForegroundModel,
  backgroundModel: BackgroundModel,
  betweenVector: Vector3 = Vector3(0.5, 8, 8)
) -> TrackingConfiguration<Tuple1<Pose2>> {
  var variableTemplate = VariableAssignments()
  var frameVariableIDs = [Tuple1<TypedID<Pose2>>]()
  for _ in 0..<frames.count {
    frameVariableIDs.append(
      Tuple1(
        variableTemplate.store(Pose2())
        ))
  }

  let addPrior = { (variables: Tuple1<TypedID<Pose2>>, values: Tuple1<Pose2>, graph: inout FactorGraph) -> () in
    let (poseID) = unpack(variables)
    let (pose) = unpack(values)
    graph.store(WeightedPriorFactorPose2(poseID, pose, weight: 1e-2, rotWeight: 2e2))
  }

  let addTrackingFactor = { (variables: Tuple1<TypedID<Pose2>>, frame: Tensor<Float>, graph: inout FactorGraph) -> () in
    let (poseID) = unpack(variables)
    graph.store(
      ProbablisticTrackingFactor(poseID,
        measurement: frame,
        encoder: model,
        patchSize: targetSize,
        appearanceModelSize: targetSize,
        foregroundModel: foregroundModel,
        backgroundModel: backgroundModel,
        maxPossibleNegativity: 1e4
      )
    )
  }

  return TrackingConfiguration(
    frames: frames,
    variableTemplate: variableTemplate,
    frameVariableIDs: frameVariableIDs,
    addPriorFactor: addPrior,
    addTrackingFactor: addTrackingFactor,
    addBetweenFactor: { (variables1, variables2, graph) -> () in
      let (poseID1) = unpack(variables1)
      let (poseID2) = unpack(variables2)
      graph.store(WeightedBetweenFactorPose2(poseID1, poseID2, Pose2(), weight: 1e-2, rotWeight: 2e2))
    },
    addFixedBetweenFactor: { (values, variables, graph) -> () in
      let (prior) = unpack(values)
      let (poseID) = unpack(variables)
      graph.store(WeightedPriorFactorPose2SD(poseID, prior, sdX: betweenVector.x, sdY: betweenVector.y, sdTheta:betweenVector.x))
    })
}

public struct ProbablisticTracker<Encoder: AppearanceModelEncoder, FG: GenerativeDensity, BG: GenerativeDensity> {
  public let encoder: Encoder
  public let foregroundModel: FG
  public let backgroundModel: BG
  public var forFrames: OISTBeeVideo
  public var trackingConfiguration: TrackingConfiguration<Tuple1<Pose2>>

  /// Colect all hyperparameters here
  public struct HyperParameters {
    public init(encoder: Encoder.HyperParameters?, foregroundModel: FG.HyperParameters? = nil, backgroundModel: BG.HyperParameters? = nil, onFrames: OISTBeeVideo, frameStatistics: FrameStatistics) {
      self.encoder = encoder
      self.foregroundModel = foregroundModel
      self.backgroundModel = backgroundModel
      self.onFrames = onFrames
      self.frameStatistics = frameStatistics
    }
    let onFrames: OISTBeeVideo
    let encoder: Encoder.HyperParameters?
    let foregroundModel: FG.HyperParameters?
    let backgroundModel: BG.HyperParameters?
    let frameStatistics: FrameStatistics
  }

  /// Initialize from three parts
  public init(encoder: Encoder, foregroundModel: FG,  backgroundModel: BG, trackingConfiguration: TrackingConfiguration<Tuple1<Pose2>>, forFrames: OISTBeeVideo) {
    self.encoder = encoder
    self.foregroundModel = foregroundModel
    self.backgroundModel = backgroundModel
    self.trackingConfiguration = trackingConfiguration
    self.forFrames = forFrames
  }

  public init(foregroundPatches: Tensor<Double>, backgroundPatches: Tensor<Double>, frames: [Tensor<Float>], given p:HyperParameters? = nil) {
    
    let trainedEncoder = Encoder(from: foregroundPatches, given: p?.encoder)
    let batchPositive = trainedEncoder.encode(foregroundPatches)
    let foregroundModel = FG(from:batchPositive, given:p?.foregroundModel)

    let batchNegative = trainedEncoder.encode(backgroundPatches)
    let backgroundModel = BG(from: batchNegative, given:p?.backgroundModel)
    let tracker = makeProbabilisticTracker(
      model: trainedEncoder,
      frames: frames, targetSize: (40, 70),
      foregroundModel: foregroundModel, backgroundModel: backgroundModel
    )
    self.init(encoder: trainedEncoder, foregroundModel: foregroundModel, backgroundModel: backgroundModel, trackingConfiguration: tracker, forFrames: p!.onFrames)
    
  }

  public mutating func infer(start: OrientedBoundingBox) -> [OrientedBoundingBox] {
    let prediction = self.trackingConfiguration.infer(knownStart: Tuple1(start.center), withSampling: true)
    let track = self.trackingConfiguration.frameVariableIDs.map { OrientedBoundingBox(center: prediction[unpack($0)], rows: 40, cols:70) }
    return track
  }

  public mutating func infer(start: OrientedBoundingBox, frames: [Tensor<Float>]) -> [OrientedBoundingBox] {
    print("\(start.center),\(frames.first!.shape)")
    var tracker = makeProbabilisticTracker(
      model: self.encoder,
      frames: frames, targetSize: (40, 70),
      foregroundModel: self.foregroundModel, backgroundModel: self.backgroundModel
    )
    
    var prediction = tracker.infer(knownStart: Tuple1(start.center), withSampling: true)
    let track = tracker.frameVariableIDs.map { OrientedBoundingBox(center: prediction[unpack($0)], rows: 40, cols:70) }
    return track
  }

  public mutating func sampleFromFactorGraph(start: OrientedBoundingBox, frames: [Tensor<Float>], numberOfSamples: Int) -> [[OrientedBoundingBox]] {
    print("\(start.center),\(frames.first!.shape)")
    var tracker = makeProbabilisticTracker(
      model: self.encoder,
      frames: frames, targetSize: (40, 70),
      foregroundModel: self.foregroundModel, backgroundModel: self.backgroundModel,
      betweenVector: Vector3(0.1, 2.0, 2.0)
    )
    
    
    var prediction = tracker.infer(knownStart: Tuple1(start.center), withSampling: true)
    var track = tracker.frameVariableIDs.map { OrientedBoundingBox(center: prediction[unpack($0)], rows: 40, cols:70) }
    var samples = [[OrientedBoundingBox]]()
    //samples.append(track)

    for _ in (0..<numberOfSamples) {
      let currentVarID = tracker.frameVariableIDs[tracker.frameVariableIDs.count - 1]
      let previousVarID = tracker.frameVariableIDs[tracker.frameVariableIDs.count - 2]
      
      var graph = tracker.graph(on: (tracker.frameVariableIDs.count - 1)..<(tracker.frameVariableIDs.count))
      tracker.addFixedBetweenFactor(prediction[previousVarID], currentVarID, &graph)
      tracker.extendBySampling(x: &prediction, fromFrame:(tracker.frameVariableIDs.count - 2), withGraph:graph, numberOfSamples: 2000)
      track = tracker.frameVariableIDs.map { OrientedBoundingBox(center: prediction[unpack($0)], rows: 40, cols:70) }
      samples.append(track)
    }
    return samples
  }

}

extension ProbablisticTracker : McEmModel {
  /// Type of patch
  public enum PatchType { case fg, bg }
  /// As datum we have a (giant) image and a noisy manual label for an image patch
  public typealias Datum = (frameID: Int, type: PatchType, obb:OrientedBoundingBox)
  /// As hidden variable we use the "true" pose of the patch
  public enum Hidden { case fg(Pose2), bg }
  /// Stack patches for all bounding boxes
  public static func patches(at regions:[Datum], given p:HyperParameters?) -> Tensor<Double> {
    return p!.frameStatistics.normalized(Tensor<Double>(regions.map { Tensor<Double>(Tensor<Float>(p!.onFrames.loadFrame($0.frameID)!).patch(at:$0.obb))} ))
  }

  /**
   Initialize with the manually labeled images
   - Parameters:
   - data: frames with and associated oriented bounding boxes
   - p: optional hyperparameters.
   */
  public init(from data:[Datum],
              given p:HyperParameters?) {
    let foregroundPatches = Self.patches(at: data.filter {$0.type == .fg}, given: p!)
    let backgroundPatches = Self.patches(at: data.filter {$0.type == .bg}, given: p!)
    self.init(foregroundPatches: foregroundPatches, backgroundPatches: backgroundPatches, frames: [Tensor<Float>(p!.onFrames.loadFrame(data.first!.frameID)!)], given:p)
  }
  
  /// version that complies, ignoring source of entropy
  public init(from data:[Datum],
              using sourceOfEntropy: inout AnyRandomNumberGenerator,
              given p:HyperParameters?) {
    self.init(from: data, given:p)
  }
  
  /// Given a datum and a model, sample from the hidden variables
  public func sample(count n:Int, for datum: Datum,
                     using sourceOfEntropy: inout AnyRandomNumberGenerator) -> [Hidden] {
    let plt = Python.import("matplotlib.pyplot")
    var mutableSelf = self
    if datum.type == .fg {
      let predictions = mutableSelf.sampleFromFactorGraph(start: datum.obb, frames: [Tensor<Float>(forFrames.loadFrame(datum.frameID)!), Tensor<Float>(forFrames.loadFrame(datum.frameID)!)], numberOfSamples: n)
      //let samples: [Pose2] = [datum.obb.center]//predictions.map{$0[1].center}
      let samples: [Pose2] = predictions.map{$0[1].center}
      for (index, sample) in samples.enumerated() {
        let (fig, axes) = plotFrameWithPatches(frame: Tensor<Float>(forFrames.loadFrame(datum.frameID)!), actual: sample, expected: datum.obb.center, firstGroundTruth: datum.obb.center)
        fig.savefig("Results/andrew01/em/andrew02_\(index).png", bbox_inches: "tight")
        plt.close("all")
      }
      return samples.map { .fg($0) }
    }
    else {
      return (0..<n).map {_ in 
      return .bg
      }
    }
    
  }
  
  /// Given an array of frames labeled with sampled poses, create a new set of patches to train from
  public init(from labeledData: [LabeledDatum], given p: HyperParameters?) {
    let data = labeledData.map {
      (label:Hidden, datum:Datum) -> Datum in
      switch label {
      case .fg(let pose):
        let obb = datum.obb
        let newOBB = OrientedBoundingBox(center: pose, rows: obb.rows, cols: obb.cols)
        return (datum.frameID, datum.type, newOBB)
      case .bg:
        return datum
      }
    }
    self.init(from: data, given:p)
  }
}

/// Returns `t` as a Swift tuple.
fileprivate func unpack<A, B>(_ t: Tuple2<A, B>) -> (A, B) {
  return (t.head, t.tail.head)
}
/// Returns `t` as a Swift tuple.
fileprivate func unpack<A>(_ t: Tuple1<A>) -> (A) {
  return (t.head)
}
