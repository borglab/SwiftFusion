import _Differentiation
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

/// Optimizes the variables in a factor graph to minimize the error, using gradient descent.
public struct GradientDescent {
  /// The fraction of the gradient to move per step.
  public var learningRate: Double
  public var baseLearningRate: Double

  /// Creates an instance with the given `learningRate`.
  public init(learningRate: Double) {
    self.learningRate = learningRate
    self.baseLearningRate = learningRate
  }
  /// Get the learning rate schedule based on the dataset size
    ///
    /// - Parameters:
    ///   - datasetSize: number of images in the current dataset
    /// - Returns: learning rate schedule based on the current dataset
    func getSchedule(datasetSize: Int) -> Array<Int> {
    if datasetSize == 100 {
        return [3, 6, 10, 100]
    }
    if datasetSize < 20000{
        return [100, 200, 300, 400, 500]
    }
    else if datasetSize < 500000 {
        return [500, 3000, 6000, 9000, 10000]
    }
    else {
        return [500, 6000, 12000, 18000, 20000]
    }
    }
  /// Get learning rate at the current step given the dataset size and base learning rate
    ///
    /// - Parameters:
    ///   - step: current training step
    ///   - datasetSize: number of images in the dataset
    ///   - baseLearningRate: starting learning rate to modify
    /// - Returns: learning rate at the current step in training
    func getLearningRate(step: Int, datasetSize: Int, baseLearningRate: Float = 0.003) -> Float? {
    let supports = getSchedule(datasetSize: datasetSize)
    // Linear warmup
    if step < supports[0] {
        return baseLearningRate * Float(step) / Float(supports[0])
    }
    // End of training
    else if step >= supports.last! {
        return nil
    }
    // Staircase decays by factor of 10
    else {
        var baseLearningRate = baseLearningRate
        for s in supports[1...] {
        if s < step {
            baseLearningRate = baseLearningRate / 10.0
        }
        }
        return baseLearningRate
    }
 }
  /// Moves `values` along the gradient of `objective`'s error function for a single gradient
  /// descent step.
  public func update(_ values: inout VariableAssignments, objective: FactorGraph) {
    values.move(along: -learningRate * objective.errorGradient(at: values))
  }
}

extension GradientDescent : Optimizer {
    public mutating func optimize(graph: FactorGraph, initial: inout VariableAssignments) {
        for i in 0..<15 {
          self.learningRate = Double(getLearningRate(step: i + 1, datasetSize: 100, baseLearningRate: Float(self.baseLearningRate))!)
          self.update(&initial, objective: graph)
        }
    } 
}