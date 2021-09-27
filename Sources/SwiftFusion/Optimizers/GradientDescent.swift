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

  /// Creates an instance with the given `learningRate`.
  public init(learningRate: Double) {
    self.learningRate = learningRate
  }

  /// Moves `values` along the gradient of `objective`'s error function for a single gradient
  /// descent step.
  public func update(_ values: inout VariableAssignments, objective: FactorGraph) {
    // print(objective.errorGradient(at: values))
    values.move(along: -learningRate * objective.errorGradient(at: values))
  }
}

extension GradientDescent : Optimizer {
    public mutating func optimize(graph: FactorGraph, initial: inout VariableAssignments) {
        // for _ in 0..<100 {
        //   self.update(&initial, objective: graph)
        // }
        print("gd doing nothing")
        // self.update(&initial, objective: graph)
    } 
}