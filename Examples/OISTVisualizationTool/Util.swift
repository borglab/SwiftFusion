

import ArgumentParser
import BeeDataset
import BeeTracking
import PenguinStructures
import PenguinParallelWithFoundation
import PythonKit
import SwiftFusion
import TensorFlow
import Foundation

/// Returns a `[[h, w, c]]` batch of normalized patches from a VOT video, and returns the
/// statistics used to normalize them.
/// - dataset: Bee video dataset object
/// - appearanceModelSize: [H, W]
/// - batchSize: number of batch samples
/// - seed: Allow controlling the random sequence
/// - trainSplit: Controls where in the frames to split between train and test
public func makeOISTTrainingBatch(dataset: OISTBeeVideo, appearanceModelSize: (Int, Int), batchSize: Int = 300, seed: Int = 42, trainSplit: Int = 250)
  -> (normalized: [Tensor<Double>], statistics: FrameStatistics)
{
  var images: [Tensor<Double>] = []
  images.reserveCapacity(batchSize)

  var currentFrame: Tensor<Double> = [0]
  var currentId: Int = -1

  var statistics = FrameStatistics(Tensor<Double>([0.0]))
  statistics.mean = Tensor(62.26806976644069)
  statistics.standardDeviation = Tensor(37.44683834503672)

  var deterministicEntropy = ARC4RandomNumberGenerator(seed: seed)
  for label in dataset.labels[0..<trainSplit].randomSelectionWithoutReplacement(k: 10, using: &deterministicEntropy).lazy.joined().randomSelectionWithoutReplacement(k: batchSize, using: &deterministicEntropy).sorted(by: { $0.frameIndex < $1.frameIndex }) {
    if currentId != label.frameIndex {
      currentFrame = Tensor<Double>(dataset.loadFrame(label.frameIndex)!)
      currentId = label.frameIndex
    }
    images.append(statistics.normalized(currentFrame.patch(at: label.location, outputSize: appearanceModelSize)))
  }

  return (images, statistics)
}


/// Returns a tracking configuration for a tracker using an RAE.
///
/// Parameter model: The RAE model to use.
/// Parameter statistics: Normalization statistics for the frames.
/// Parameter frames: The frames of the video where we want to run tracking.
/// Parameter targetSize: The size of the target in the frames.
public func makeNaiveBayesRAETracker(
  model: DenseRAE,
  statistics: FrameStatistics,
  frames: [Tensor<Float>],
  targetSize: (Int, Int),
  foregroundModel: MultivariateGaussian,
  backgroundModel: GaussianNB
) -> TrackingConfiguration<Tuple1<Pose2>> {
  var variableTemplate = VariableAssignments()
  var frameVariableIDs = [Tuple1<TypedID<Pose2>>]()
  for _ in 0..<frames.count {
    frameVariableIDs.append(
      Tuple1(
        variableTemplate.store(Pose2())
        ))
  }
  return TrackingConfiguration(
    frames: frames,
    variableTemplate: variableTemplate,
    frameVariableIDs: frameVariableIDs,
    addPriorFactor: { (variables, values, graph) -> () in
      let (poseID) = unpack(variables)
      let (pose) = unpack(values)
      graph.store(WeightedPriorFactor(poseID, pose, weight: 1e-1))
    },
    addTrackingFactor: { (variables, frame, graph) -> () in
      let (poseID) = unpack(variables)
      graph.store(
        ProbablisticTrackingFactor(poseID,
          measurement: statistics.normalized(frame),
          encoder: model,
          patchSize: targetSize,
          appearanceModelSize: targetSize,
          foregroundModel: foregroundModel,
          backgroundModel: backgroundModel,
          maxPossibleNegativity: 1e1
        )
      )
    },
    addBetweenFactor: { (variables1, variables2, graph) -> () in
      let (poseID1) = unpack(variables1)
      let (poseID2) = unpack(variables2)
      graph.store(WeightedBetweenFactorPose2(poseID1, poseID2, Pose2(), weight: 1e-1, rotWeight: 1e2))
    })
}

/// Returns a tracking configuration for a tracker using PCA.
///
/// Parameter model: The PCA model to use.
/// Parameter statistics: Normalization statistics for the frames.
/// Parameter frames: The frames of the video where we want to run tracking.
/// Parameter targetSize: The size of the target in the frames.
public func makeNaiveBayesPCATracker(
  model: PPCA,
  statistics: FrameStatistics,
  frames: [Tensor<Float>],
  targetSize: (Int, Int),
  foregroundModel: MultivariateGaussian,
  backgroundModel: GaussianNB
) -> TrackingConfiguration<Tuple1<Pose2>> {
  var variableTemplate = VariableAssignments()
  var frameVariableIDs = [Tuple1<TypedID<Pose2>>]()
  for _ in 0..<frames.count {
    frameVariableIDs.append(
      Tuple1(
        variableTemplate.store(Pose2())
        ))
  }
  return TrackingConfiguration(
    frames: frames,
    variableTemplate: variableTemplate,
    frameVariableIDs: frameVariableIDs,
    addPriorFactor: { (variables, values, graph) -> () in
      let (poseID) = unpack(variables)
      let (pose) = unpack(values)
      graph.store(WeightedPriorFactor(poseID, pose, weight: 1e-1))
    },
    addTrackingFactor: { (variables, frame, graph) -> () in
      let (poseID) = unpack(variables)
      graph.store(
        ProbablisticTrackingFactor(poseID,
          measurement: statistics.normalized(frame),
          encoder: model,
          patchSize: targetSize,
          appearanceModelSize: targetSize,
          foregroundModel: foregroundModel,
          backgroundModel: backgroundModel,
          maxPossibleNegativity: 1e5
        )
      )
    },
    addBetweenFactor: { (variables1, variables2, graph) -> () in
      let (poseID1) = unpack(variables1)
      let (poseID2) = unpack(variables2)
      graph.store(WeightedBetweenFactorPose2(poseID1, poseID2, Pose2(), weight: 1e-1, rotWeight: 1e2))
    })
}

/// Returns `t` as a Swift tuple.
fileprivate func unpack<A, B>(_ t: Tuple2<A, B>) -> (A, B) {
  return (t.head, t.tail.head)
}
/// Returns `t` as a Swift tuple.
fileprivate func unpack<A>(_ t: Tuple1<A>) -> (A) {
  return (t.head)
}
