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

import TensorFlow
import XCTest

/// Use low-variance resampling method (rake method) to resample, with replacement `count`samples from `samples`
public func resample<T, G: RandomNumberGenerator>(count n:Int, from samples: [(weight:Double,element:T)], using generator: inout G) -> [T] {
  // Calculate total weight.
  let sum = samples.map(\.weight).reduce(0, +)
  
  // Sample position of first rake spike.
  let delta = sum/Double(n)
  var position = Double.random(in: 0...delta, using: &generator)
  
  // Create a sample for each rake spike.
  var index = 0
  var cumulative = samples[index].weight
  let result = (0..<n).map { _ -> T in
    // Skip over samples until weight interval for index contains rake position.
    while position > cumulative {
      index += 1
      cumulative += samples[index].weight
    }
    // Update rake position and return sample element.
    position += delta
    return samples[index].element
  }
  
  return result
}

public func resample<T>(count n:Int, from samples: [(Double,T)]) -> [T] {
  var g = SystemRandomNumberGenerator()
  return resample(count:n, from:samples, using: &g)
}

/// Protocol for models that can be fitted with Monte Carle EM
public protocol McEmModel {
  associatedtype Datum /// main data a model is trained with
  associatedtype Hidden /// type for hidden variable associated with a Datum
  associatedtype HyperParameters
  typealias LabeledDatum = (Hidden, Datum)

  init(from data: [Datum],
       using sourceOfEntropy: inout AnyRandomNumberGenerator,
       given: HyperParameters?)
  func sample(count:Int,
              for datum: Datum,
              using sourceOfEntropy: inout AnyRandomNumberGenerator) -> [Hidden]
  init(from labeledData: [LabeledDatum], given: HyperParameters?)
}

public extension McEmModel {
  /// Extension allows to have a default nil parameter
  init(from data: [Datum], using sourceOfEntropy: inout AnyRandomNumberGenerator) {
    self.init(from: data, using: &sourceOfEntropy, given: nil)
  }
  init(from labeledData: [LabeledDatum]) {
    self.init(from: labeledData, given: nil)
  }
}

public extension McEmModel {
  /// Extension allows to have a default nil parameter
  init(_ data: [Datum],
       using sourceOfEntropy: inout AnyRandomNumberGenerator,
       given: HyperParameters? = nil) {
    self.init(from: data, using: &sourceOfEntropy, given: nil)
  }
}

/// Monte Carlo EM algorithm
public struct MonteCarloEM<ModelType: McEmModel> {
  public typealias Hook = (Int, [ModelType.LabeledDatum], ModelType) -> ()
  var sourceOfEntropy: AnyRandomNumberGenerator

  /// Initialize, possibly witha random number generator
  public init(sourceOfEntropy: RandomNumberGenerator = SystemRandomNumberGenerator()) {
    self.sourceOfEntropy = .init(sourceOfEntropy)
  }
  
  /// Run Monte Carlo EM given unlabeled data
  public mutating func run(with data:[ModelType.Datum],
                           modelInitializer: (_ data: [ModelType.Datum], _ entropySource: inout AnyRandomNumberGenerator) -> ModelType,
                           modelFitter: (_ data: [ModelType.LabeledDatum]) -> ModelType,
                           iterationCount:Int,
                           sampleCount:Int = 5,
                           hook: Hook? = {(_,_,_) in () }) -> ModelType {
    var model = modelInitializer(data, &self.sourceOfEntropy)
    
    for i in 1...iterationCount {
      // Monte-Carlo E-step: given current model, sample hidden variables for each datum
      var labeledData = [ModelType.LabeledDatum]()
      for datum in data {
        // Given a datum and a model, sample from the hidden variables
        let sample = model.sample(count: sampleCount, for: datum, using: &self.sourceOfEntropy)
        labeledData.append(contentsOf: sample.map { ($0, datum) })
      }
      
      // M-step: fit model using labeled datums
      model = modelFitter(labeledData)
      
      // Call hook if given
      hook?(i, labeledData, model)
    }
    return model
  }
  
  /// Run Monte Carlo EM given unlabeled data
  public mutating func run(with data:[ModelType.Datum],
                           iterationCount:Int,
                           sampleCount:Int = 5,
                           hook: Hook? = {(_,_,_) in () }) -> ModelType {
    let modelInitializer = { ModelType(from: $0, using: &$1) }
    let modelFitter = { ModelType(from: $0) }
    
    let model = run(with: data, modelInitializer: modelInitializer, modelFitter: modelFitter, iterationCount: iterationCount, sampleCount: sampleCount, hook: hook)
    return model
  }
}

