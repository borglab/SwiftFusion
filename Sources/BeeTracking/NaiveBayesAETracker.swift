// Copyright 2020 The SwiftFusion Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import SwiftFusion
import TensorFlow
import PenguinStructures

/// Returns a tracking configuration for a tracker using an RAE.
///
/// Parameter model: The RAE model to use.
/// Parameter statistics: Normalization statistics for the frames.
/// Parameter frames: The frames of the video where we want to run tracking.
/// Parameter targetSize: The size of the target in the frames.
public func makeNaiveBayesAETracker(
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
      graph.store(WeightedPriorFactorPose2(poseID, pose, weight: 1e0, rotWeight: 1e2))
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
      graph.store(WeightedBetweenFactorPose2SD(poseID1, poseID2, Pose2(), sdX: 8, sdY: 4.6, sdTheta: 0.3))
    },
    addFixedBetweenFactor: { (values, variables, graph) -> () in
      let (prior) = unpack(values)
      let (poseID) = unpack(variables)
      graph.store(WeightedPriorFactorPose2SD(poseID, prior, sdX: 8, sdY: 4.6, sdTheta: 0.3))
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
