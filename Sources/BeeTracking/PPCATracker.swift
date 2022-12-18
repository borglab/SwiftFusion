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
// import TensorFlow
import PenguinStructures

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
