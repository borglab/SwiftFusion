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

/// A factor graph for linear problems
/// Factors are the Jacobians between the corresponding variables and measurements
/// TODO(fan): Add noise model
public struct NonlinearFactorGraph: FactorGraph {
  public typealias KeysType = Array<Int>
  
  public typealias FactorsType = Array<AnyNonlinearFactor>
  
  public var keys: KeysType = []
  public var factors: FactorsType = []
  
  /// Default initializer
  public init() { }
  
  /// Convenience operator for adding factor
  public static func += (lhs: inout Self, rhs: AnyNonlinearFactor) {
    lhs.factors.append(rhs)
  }
  
  /// linearize the nonlinear factor graph to a linear factor graph
  public func linearize(_ values: Values) -> GaussianFactorGraph {
    var gfg = GaussianFactorGraph()
    
    for i in factors {
      let linearized = i.linearize(values)
      
      // Assertion for the shape of Jacobian
      assert(linearized.jacobians.map { $0.shape.count == 2 }.reduce(true, { $0 && $1 }))
      
      gfg += linearized
    }
    
    return gfg
  }
}
