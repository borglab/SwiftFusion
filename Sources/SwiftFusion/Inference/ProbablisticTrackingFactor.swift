// Copyright 2019 The SwiftFusion Authors. All Rights Reserved.
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

import PenguinParallel
import PenguinStructures
import TensorFlow

/// A factor over a target's pose and appearance in an image.
public struct ProbablisticTrackingFactor<
    Encoder: AppearanceModelEncoder,
    ForegroundModel: GenerativeDensity, BackgroundModel: GenerativeDensity
  >: LinearizableFactor1 {
  /// The first adjacent variable, the pose of the target in the image.
  ///
  /// This explicitly specifies `LinearizableFactor2`'s `associatedtype V0`.
  public typealias V0 = Pose2

  /// The IDs of the variables adjacent to this factor.
  public let edges: Variables.Indices

  /// The image containing the target.
  public let measurement: Tensor<Double>

  public let encoder: Encoder

  public var patchSize: (Int, Int)

  public var appearanceModelSize: (Int, Int)

  public var foregroundModel: ForegroundModel

  public var backgroundModel: BackgroundModel
  /// Creates an instance.
  ///
  /// - Parameters:
  ///   - poseId: the id of the adjacent pose variable.
  ///   - measurement: the image containing the target.
  ///   - appearanceModel: the generative model that produces an appearance from a latent code.
  ///   - foregroundModel: A generative density on the foreground
  ///   - backgroundModel: A generative density on the background
  public init(
    _ poseId: TypedID<Pose2>,
    measurement: Tensor<Double>,
    encoder: Encoder,
    patchSize: (Int, Int),
    appearanceModelSize: (Int, Int),
    foregroundModel: ForegroundModel,
    backgroundModel: BackgroundModel
  ) {
    self.edges = Tuple1(poseId)
    self.measurement = measurement
    self.encoder = encoder
    self.patchSize = patchSize
    self.appearanceModelSize = appearanceModelSize
    self.foregroundModel = foregroundModel
    self.backgroundModel = backgroundModel
  }

  @differentiable
  public func errorVector(_ pose: Pose2) -> Vector1 {
    let region = OrientedBoundingBox(center: pose, rows: patchSize.0, cols: patchSize.1)
    let patch = measurement.patch(at: region, outputSize: appearanceModelSize)
    let features = encoder.encode(patch)

    return Vector1(foregroundModel.negativeLogLikelihood(features) - backgroundModel.negativeLogLikelihood(features))
  }
}
