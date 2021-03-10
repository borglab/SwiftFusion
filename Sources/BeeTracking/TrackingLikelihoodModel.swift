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
import SwiftFusion
import TensorFlow
import XCTest
import PythonKit

/// A likelihood model has a a feature encoder, a foreground model and a background model
public struct TrackingLikelihoodModel<Encoder: AppearanceModelEncoder, FG:GenerativeDensity, BG:GenerativeDensity> {
  public let encoder: Encoder
  public let foregroundModel: FG
  public let backgroundModel: BG
  public let frameStatistics: FrameStatistics
  
  /// Colect all hyperparameters here
  public struct HyperParameters {
    public init(encoder: Encoder.HyperParameters?, foregroundModel: FG.HyperParameters? = nil, backgroundModel: BG.HyperParameters? = nil, frameStatistics: FrameStatistics? = nil) {
      self.encoder = encoder
      self.foregroundModel = foregroundModel
      self.backgroundModel = backgroundModel
      self.frameStatistics = frameStatistics
    }
    let frameStatistics: FrameStatistics?
    let encoder: Encoder.HyperParameters?
    let foregroundModel: FG.HyperParameters?
    let backgroundModel: BG.HyperParameters?
  }
  
  /// Initialize from three parts
  public init(encoder: Encoder, foregroundModel: FG,  backgroundModel: BG, frameStatistics: FrameStatistics? = nil) {
    self.encoder = encoder
    self.foregroundModel = foregroundModel
    self.backgroundModel = backgroundModel
    self.frameStatistics = frameStatistics!
  }
  
  /**
   Train all models from a collection of foreground and background patches
   - Parameters:
    - foregroundPatches: [...,H,W,C] batch of foreground patches to train with
    - backgroundPatches: [...,H,W,C] batch of background patches to train with
    - p: optional hyperparameters.
   */
  public init(from foregroundPatches: Tensor<Double>,
              and backgroundPatches: Tensor<Double>,
              given p:HyperParameters? = nil) {
    let trainedEncoder = Encoder(from: foregroundPatches, given: p?.encoder)
    let fgFeatures = trainedEncoder.encode(foregroundPatches)
    let bgFeatures = trainedEncoder.encode(backgroundPatches)
    self.init(encoder: trainedEncoder,
              foregroundModel : FG(from: fgFeatures, given:p?.foregroundModel),
              backgroundModel : BG(from: bgFeatures, given:p?.backgroundModel),
              frameStatistics: p!.frameStatistics)
  }

  /// Calculate the negative log likelihood of the likelihood model of a patch
  @differentiable
  public func negativeLogLikelihood(of patch: Tensor<Double>) -> Double {
    let encoded = encoder.encode(sample: patch)
    return (foregroundModel.negativeLogLikelihood(encoded) - backgroundModel.negativeLogLikelihood(encoded))
  }
  
  /// Calculate the probability of the likelihood model of a patch
  @differentiable public func unnormalizedProbability(of patch: Tensor<Double>) -> Double {
    let E = negativeLogLikelihood(of: patch) + 1e6
    return exp(-E)
  }
}

extension TrackingLikelihoodModel : McEmModel {
  /// Type of patch
  public enum PatchType { case fg, bg }
  
  /// As datum we have a (giant) image and a noisy manual label for an image patch
  public typealias Datum = (frame: Tensor<Double>?, type: PatchType, obb:OrientedBoundingBox)
  
  /// As hidden variable we use the "true" pose of the patch
  public enum Hidden { case fg(Pose2), bg }
  
  /// Stack patches for all bounding boxes
  public static func patches(at regions:[Datum], given p:HyperParameters?) -> Tensor<Double> {
    return p!.frameStatistics!.normalized(Tensor<Double>(regions.map { $0.frame!.patch(at:$0.obb) } ))
  }

  /**
   Initialize with the manually labeled images
   - Parameters:
   - data: frames with and associated oriented bounding boxes
   - p: optional hyperparameters.
   */
  public init(from data:[Datum],
              given p:HyperParameters?) {
    let foregroundPatches = Self.patches(at: data.filter {$0.type == .fg}, given: p)
    let backgroundPatches = Self.patches(at: data.filter {$0.type == .bg}, given: p)
    self.init(from: foregroundPatches, and: backgroundPatches, given:p)
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
    // We first approximate the posterior over foreground poses using importance sampling:
    let samples : [(Double, Pose2)] = (0..<5).map { index in
      // sample from noise model on manual pose
      var proposal = datum.obb.center
      proposal.perturbWith(stddev: Vector3(0.1, 2, 2))
      let patch = self.frameStatistics.normalized(datum.frame!.patch(
        at: OrientedBoundingBox(
          center: proposal, rows: datum.obb.rows, cols: datum.obb.cols
        )
      ))
      // let plt = Python.import("matplotlib.pyplot")
      // let (fig, axes) = plotFrameWithPatches(frame: Tensor<Float>(datum.frame!), actual: proposal, expected: datum.obb.center, firstGroundTruth: datum.obb.center)
      // fig.savefig("Results/andrew01/em/andrew02_\(index).png", bbox_inches: "tight")
      // plt.close("all")
      
      return (
        // The weight of the proposal, which equals to the unnormalized probability of the likelihood:
        self.unnormalizedProbability(of: patch), proposal
      )
    }
    
    // Then resample to get unweighted samples:
    let resampled = resample(count:n, from:samples, using: &sourceOfEntropy)
    // let weights = samples.map{ $0.0 }
    // let maxWeightIdx = weights.index(of: weights.max()!)!
    // let resampled = [samples[maxWeightIdx]]
    // Finally, return as foreground labels:
    return resampled.map { .fg($0) }
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
