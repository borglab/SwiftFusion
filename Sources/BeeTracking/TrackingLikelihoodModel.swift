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

/// A likelihood model has a a feature encoder, a foreground model and a background model
public struct TrackingLikelihoodModel<Encoder: AppearanceModelEncoder, FG:GenerativeDensity, BG:GenerativeDensity> {
  public let encoder: Encoder
  public let foregroundModel: FG
  public let backgroundModel: BG
  
  /// Colect all hyperparameters here
  public struct HyperParameters {
    let encoder: Encoder.HyperParameters
    let foregroundModel: FG.HyperParameters
    let backgroundModel: BG.HyperParameters
    
    /// Part
    let backgroundPatches: Tensor<Double>
  }
  
  /// Initialize from three parts
  public init(encoder: Encoder, foregroundModel: FG,  backgroundModel: BG) {
    self.encoder = encoder
    self.foregroundModel = foregroundModel
    self.backgroundModel = backgroundModel
  }
  
  /**
   Train encoder and foreground model from a foreground patches
   - Parameters:
   - fgPatches: [...,H,W,C] batch of foreground patches to train with
   - backgroundModel: assumed to be trained separately
   - p: optional hyperparameters.
   */
  public init(from fgPatches: Tensor<Double>, and backgroundModel: BG,
              given p:HyperParameters? = nil) {
    let trainedEncoder = Encoder(from: fgPatches, given: p?.encoder)
    let fgFeatures = trainedEncoder.encode(fgPatches)
    self.init(encoder: trainedEncoder,
              foregroundModel : FG(from: fgFeatures, given:p?.foregroundModel),
              backgroundModel : backgroundModel)
  }
  
  /**
   Train all models from a collection of foreground and background patches
   - Parameters:
   - fgPatches: [...,H,W,C] batch of foreground patches to train with
   - backgroundModel: assumed to be trained separately
   - p: optional hyperparameters.
   */
  public init(from fgPatches: Tensor<Double>, and bgPatches: Tensor<Double>,              given p:HyperParameters? = nil) {
    let trainedEncoder = Encoder(from: fgPatches, given: p?.encoder)
    let fgFeatures = trainedEncoder.encode(fgPatches)
    let bgFeatures = trainedEncoder.encode(bgPatches)
    self.init(encoder: trainedEncoder,
              foregroundModel : FG(from: fgFeatures, given:p?.foregroundModel),
              backgroundModel : BG(from: bgFeatures, given:p?.backgroundModel))
  }
}

/// Make it trainable with Monte Carlo EM
extension TrackingLikelihoodModel : McEmModel {
  /// As datum we have a (giant) image and a noisy manual label for an image patch
  public typealias Datum = (frame: Tensor<Double>, manualLabel:OrientedBoundingBox)
  
  /// As hidden variable we use the "true" pose of the patch
  public typealias Hidden = Pose2
  
  /**
   Initialize with the manually labeled images
   - Parameters:
   - data: frames with and associated oriented bounding boxes
   - sourceOfEntropy: random number generator
   - p: optional hyperparameters.
   */
  public init(from data:[Datum],
              using sourceOfEntropy: inout AnyRandomNumberGenerator,
              given p:HyperParameters?) {
    let patches = data.map { $0.frame.patch(at: $0.manualLabel) }
    let imageBatch = Tensor<Double>(patches)
    let backgroundModel = BG(from: imageBatch)
    self.init(from: imageBatch, and: backgroundModel, given:p)
  }
  
  /// Given a datum and a model, sample from the hidden variables
  public func sample(count:Int, for datum: Datum,
                     using sourceOfEntropy: inout AnyRandomNumberGenerator) -> [Hidden] {
    // Two approaches: importance sampling, OR, what we do here:
    // First optimize pose using LM, then sample from pose covariance around minimum
    let labels = [Hidden]()
    //    let labels : [Hidden] = (0..<count).map { _ in
    //      let u = Double.random(in: 0..<p1+p2, using: &sourceOfEntropy)
    //      return u<=p1 ? .one : .two
    //    }
    return labels
  }
  
  /// Given an array of labeled datums, fit the two Gaussian mixture components
  mutating public func fit(_ labeledData: [LabeledDatum]) {
  }
}
