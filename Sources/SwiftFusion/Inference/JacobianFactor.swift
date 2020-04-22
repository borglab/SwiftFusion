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

/// A `LinearFactor` that operates like `JacobianFactor` in GTSAM.
///
/// Input is a dictionary of `Key` to `Tensor` pairs, and the output is the paired
/// error vector. Note here that the Tensor shapes are not checked.
///
/// Interpretation
/// ================
/// `Input`: the input values as key-value pairs
/// `Errors`: the vector `J * x`, where `J` is `m*n` and `x` is `n*1` and returns a m vector
///
/// In block form, it looks like the following:
/// ```
///                 x1
/// [ J1 J2 J3 ] *  x2 = [ J1*x1 J2*x2 J3*x3 ]
///                 x3
/// ```
///
/// Explanation
/// ================
/// TODO:
/// I think both Jacobian and Hessian factors in GTSAM are converted into JacobianFactor in
/// `GaussianFactorGraph`, so it becomes a question whether we should do the same?
/// I am considering making `JacobianLikeFactor` a protocol and make `JacobianFactor`
/// and `HessianFactor` conform to this protocol instead.
public struct JacobianFactor: LinearFactor {
  
  // TODO(fan): correct this and add a unit test
  public func error(_ values: VectorValues) -> ScalarType {
    ScalarType.zero
  }
  
  public var dimension: Int {
    get {
      jacobians[0].shape.dimensions[0]
    }
  }
  public var keys: Array<Int> = []
  public var jacobians: Array<Tensor<Double>> = []
  public var b: Tensor<Double> = eye(rowCount: 0)
  public typealias Output = Error
  
  public init (_ key: [Int], _ A: [Tensor<Double>], _ b: Tensor<Double>) {
    keys = key
    jacobians = A
    self.b = b
  }
  
  /// Calculate `J*x`
  /// Comparable to the `*` operator in GTSAM
  /// In block form, it looks like the following:
  /// ```
  ///                 x1
  /// [ J1 J2 J3 ] *  x2 = [ J1*x1 J2*x2 J3*x3 ]
  ///                 x3
  /// ```
  static func * (lhs: JacobianFactor, rhs: VectorValues) -> Self.Output {
    var arr: Array<Tensor<Double>> = []
    for i in lhs.keys.indices {
      arr.append(matmul(lhs.jacobians[i], rhs[lhs.keys[i]]))
    }
    return arr.reduce(Tensor<Double>(repeating: 0.0, shape: TensorShape([lhs.dimension, 1])), { $0 + $1 })
  }
  
  /// Calculate `J^T * e`
  /// `J^T` is `n*m` and e is `m*1`
  /// In block form, it looks like the following:
  /// ```
  /// J1^T          J1Te
  /// J2^T  *  e =  J2Te
  /// J3^T          J3Te
  /// ```
  /// However, there exists the possibility that one Value is used multiple times
  /// At that time we need to add the corresponding blocks
  public func atr (_ r: Self.Output) -> VectorValues {
    var result = VectorValues()
    
    // No noise model yet
    // if (model_) model_->whitenInPlace(E);
    
    // Just iterate over all A matrices and insert Ai^e into VectorValues
    for pos in 0..<keys.count {
      let k = keys[pos]
      
      // TODO(fan): add a proper method for searching key
      if let ind = result._indices[k] {
        result._values[ind] += matmul(jacobians[pos].transposed(), r)
      } else {
        result.insert(k, matmul(jacobians[pos].transposed(), r))
      }
    }
    return result
  }
}
