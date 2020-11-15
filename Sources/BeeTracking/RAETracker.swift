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

/// Returns `t` as a Swift tuple.
fileprivate func unpack<A, B>(_ t: Tuple2<A, B>) -> (A, B) {
  return (t.head, t.tail.head)
}
/// Returns `t` as a Swift tuple.
fileprivate func unpack<A>(_ t: Tuple1<A>) -> (A) {
  return (t.head)
}
