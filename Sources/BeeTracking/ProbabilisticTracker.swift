import BeeDataset
import PenguinStructures
import SwiftFusion
import TensorFlow
import PythonKit
import Foundation

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
  backgroundModel: BackgroundModel
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
      graph.store(WeightedPriorFactorPose2SD(poseID, prior, sdX: 8, sdY: 8, sdTheta:0.4))
    })
}

public struct ProbablisticTracker<Encoder: AppearanceModelEncoder, FG: GenerativeDensity, BG: GenerativeDensity> {
  public let encoder: Encoder
  public let foregroundModel: FG
  public let backgroundModel: BG
  public var trackingConfiguration: TrackingConfiguration<Tuple1<Pose2>>

  /// Colect all hyperparameters here
  public struct HyperParameters {
    public init(encoder: Encoder.HyperParameters?, foregroundModel: FG.HyperParameters? = nil, backgroundModel: BG.HyperParameters? = nil) {
      self.encoder = encoder
      self.foregroundModel = foregroundModel
      self.backgroundModel = backgroundModel
    }
    
    let encoder: Encoder.HyperParameters?
    let foregroundModel: FG.HyperParameters?
    let backgroundModel: BG.HyperParameters?
  }

  /// Initialize from three parts
  public init(encoder: Encoder, foregroundModel: FG,  backgroundModel: BG, trackingConfiguration: TrackingConfiguration<Tuple1<Pose2>>) {
    self.encoder = encoder
    self.foregroundModel = foregroundModel
    self.backgroundModel = backgroundModel
    self.trackingConfiguration = trackingConfiguration
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
    self.init(encoder: trainedEncoder, foregroundModel: foregroundModel, backgroundModel: backgroundModel, trackingConfiguration: tracker)
    
  }

  public mutating func infer(start: OrientedBoundingBox) -> [OrientedBoundingBox] {
    let prediction = self.trackingConfiguration.infer(knownStart: Tuple1(start.center), withSampling: true)
    let track = self.trackingConfiguration.frameVariableIDs.map { OrientedBoundingBox(center: prediction[unpack($0)], rows: 40, cols:70) }
    return track
  }

  public mutating func infer(start: OrientedBoundingBox, frames: [Tensor<Float>]) -> [OrientedBoundingBox] {
    var tracker = makeProbabilisticTracker(
      model: self.encoder,
      frames: frames, targetSize: (40, 70),
      foregroundModel: self.foregroundModel, backgroundModel: self.backgroundModel
    )
    let prediction = tracker.infer(knownStart: Tuple1(start.center), withSampling: true)
    let track = tracker.frameVariableIDs.map { OrientedBoundingBox(center: prediction[unpack($0)], rows: 40, cols:70) }
    return track
  }

}

extension ProbablisticTracker : McEmModel {
  /// Type of patch
  public enum PatchType { case fg, bg }
  
  /// As datum we have a (giant) image and a noisy manual label for an image patch
  public typealias Datum = (frame: Tensor<Double>?, type: PatchType, obb:OrientedBoundingBox)
  
  /// As hidden variable we use the "true" pose of the patch
  public enum Hidden { case fg(Pose2), bg }
  
  /// Stack patches for all bounding boxes
  public static func patches(at regions:[Datum]) -> Tensor<Double> {
    return Tensor<Double>(regions.map { $0.frame!.patch(at:$0.obb) } )
  }

  /**
   Initialize with the manually labeled images
   - Parameters:
   - data: frames with and associated oriented bounding boxes
   - p: optional hyperparameters.
   */
  public init(from data:[Datum],
              given p:HyperParameters?) {
    let foregroundPatches = Self.patches(at: data.filter {$0.type == .fg})
    let backgroundPatches = Self.patches(at: data.filter {$0.type == .bg})
    self.init(foregroundPatches: foregroundPatches, backgroundPatches: backgroundPatches, frames: [Tensor<Float>(data.first!.frame!)], given:p)
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
    var mutableSelf = self
    if datum.type == .fg {
      let samples: [Pose2] = (0..<n).map {_ in 
      return mutableSelf.infer(start: datum.obb, frames: [Tensor<Float>(datum.frame!), Tensor<Float>(datum.frame!)])[1].center
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
        return (datum.frame, datum.type, newOBB)
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
