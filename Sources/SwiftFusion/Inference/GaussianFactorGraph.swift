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
public struct OldGaussianFactorGraph {
  public typealias KeysType = Array<Int>
  
  public typealias FactorsType = Array<OldJacobianFactor>
  
  public var keys: KeysType = []
  public var factors: FactorsType = []
  
  public var b: Errors {
    get {
      factors.map { $0.b }
    }
  }
  
  /// Default initializer
  public init() { }
  
  /// This calculates `A*x`, where x is the collection of key-values
  public static func * (lhs: OldGaussianFactorGraph, rhs: VectorValues) -> Errors {
    Array(lhs.factors.map { $0 * rhs })
  }
  
  /// This calculates `A*x - b`, where x is the collection of key-values
  public func residual (_ val: VectorValues) -> Errors {
    Array(self.factors.map { $0 * val - $0.b })
  }
  
  /// Convenience operator for adding factor
  public static func += (lhs: inout Self, rhs: OldJacobianFactor) {
    lhs.factors.append(rhs)
  }
  
  /// This calculates `A^T * r`, where r is the residual (error)
  public func atr(_ r: Errors) -> VectorValues {
    var vv = VectorValues()
    for i in r.indices {
      let JTr = factors[i].atr(r[i])
      
      vv = vv + JTr
    }
    
    return vv
  }
}
