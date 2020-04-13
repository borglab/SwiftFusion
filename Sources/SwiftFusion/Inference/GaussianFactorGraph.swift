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
public struct GaussianFactorGraph: FactorGraph {
  public typealias KeysType = Array<Int>
  
  public typealias FactorsType = Array<JacobianFactor>
  
  public var keys: KeysType
  public var factors: FactorsType
  
  public var b: Errors
  
  /// This calculates `A*x`, where x is the collection of key-values
  /// Note A is a
  static func * (lhs: GaussianFactorGraph, rhs: VectorValues) -> Errors {
    Array(lhs.factors.map { $0 * rhs })
  }
  
  /// This calculates `A^T * r`, where r is the residual (error)
  func atr(r: Errors) -> VectorValues {
    // Array(
    fatalError()
  }
}
