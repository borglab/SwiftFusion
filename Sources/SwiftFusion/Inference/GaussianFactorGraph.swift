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
public struct GaussianFactorGraph {
  public typealias KeysType = Array<Int>
  
  public typealias FactorsType = Array<JacobianFactor>
  
  public var keys: KeysType = []
  public var factors: FactorsType = []
  
  public var b: Errors {
    get {
      Errors(factors.map { $0.b })
    }
  }
  
  /// Default initializer
  public init() { }
  
  /// This calculates `A*x`, where x is the collection of key-values
  public static func * (lhs: GaussianFactorGraph, rhs: VectorValues) -> Errors {
    return Errors(lhs.factors.map { $0 * rhs })
  }
  
  /// This calculates `A*x - b`, where x is the collection of key-values
  public func residual (_ val: VectorValues) -> Errors {
    return Errors(self.factors.map { $0 * val - $0.b })
  }
  
  /// Convenience operator for adding factor
  public static func += (lhs: inout Self, rhs: JacobianFactor) {
    lhs.factors.append(rhs)
  }
  
  /// This calculates `A^T * r`, where r is the residual (error)
  public func atr(_ r: Errors) -> VectorValues {
    var vv = VectorValues()
    for i in r.values.indices {
      let JTr = factors[i].atr(r.values[i])
      
      vv = vv + JTr
    }
    
    return vv
  }
}

extension GaussianFactorGraph: DecomposedAffineFunction {
  public typealias Input = VectorValues
  public typealias Output = Errors

  public func callAsFunction(_ x: Input) -> Output {
    return residual(x)
  }

  public func applyLinearForward(_ x: Input) -> Output {
    return self * x
  }

  public func applyLinearAdjoint(_ y: Output) -> Input {
    return self.atr(y)
  }

  public var bias: Output {
    return b.scaled(by: -1)
  }
}
