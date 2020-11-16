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

/// Monte Carlo EM algorithm
public struct MonteCarloEM<ModelType: McEmModel> {
  public typealias Hook = (Int, [ModelType.LabeledDatum], ModelType) -> ()
  var sourceOfEntropy: AnyRandomNumberGenerator
  
  /// Initialize, possibly witha random number generator
  init(sourceOfEntropy: RandomNumberGenerator = SystemRandomNumberGenerator()) {
    self.sourceOfEntropy = .init(sourceOfEntropy)
  }
  
  /// Run Monte Carlo EM given unlabeled data
  public mutating func run(with data:[ModelType.Datum],
                           iterationCount:Int,
                           sampleCount:Int = 5,
                           hook: Hook? = {(_,_,_) in () }) -> ModelType {
    var model = ModelType(from: data, using: &self.sourceOfEntropy)
    
    for i in 1...iterationCount {
      // Monte-Carlo E-step: given current model, sample hidden variables for each datum
      var labeledData = [ModelType.LabeledDatum]()
      for datum in data {
        // Given a datum and a model, sample from the hidden variables
        let sample = model.sample(count: sampleCount, for: datum, using: &self.sourceOfEntropy)
        labeledData.append(contentsOf: sample.map { ($0, datum) })
      }
      
      // M-step: fit model using labeled datums
      model = ModelType(from: labeledData)
      
      // Call hook if given
      hook?(i, labeledData, model)
    }
    return model
  }
}

