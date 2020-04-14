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

//public protocol Value {
//  
//}

public struct VectorValues {
  typealias ScalarType = Double
  var _values: [Tensor<ScalarType>]
  var _indices: Dictionary<Int, Int>
  
  /// The subscript operator, with some indirection
  /// Should be replaced after Dictionary is in
  subscript(key: Int) -> Tensor<ScalarType> {
    _values[_indices[key]!]
  }
  
  /// L2 norm of the VectorValues
  var norm: Double {
    self._values.map { $0.squared().sum().scalar! }.reduce(0.0, { $0 + $1 })
  }
  
  static func + (_ lhs: Self, _ rhs: Self.ScalarType) -> Self {
    var result = lhs
    let _ = result._values.indices.map { result._values[$0] += rhs }
    return result
  }
  
  static func + (_ lhs: Self, _ rhs: Self) -> Self {
    var result = lhs
    for (k, i_r) in rhs._indices {
      if let i_l = lhs._indices[k] {
        result._values[i_l] += rhs._values[i_r]
      } else {
        result._indices[k] = result._values.count
        result._values.append(rhs._values[i_r])
      }
    }
    return result
  }
  
  static func * (_ lhs: Self.ScalarType, _ rhs: Self) -> Self {
    var result = rhs
    let _ = result._values.indices.map { result._values[$0] *= lhs }
    return result
  }
}
