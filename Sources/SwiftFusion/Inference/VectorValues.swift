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

/// The class that holds Key-Vector pairs.
public struct VectorValues: KeyPathIterable {
  public typealias ScalarType = Double
  var _values: [Tensor<ScalarType>] = []
  
  /// Dictionary from Key to index
  var _indices: Dictionary<Int, Int> = [:]
  
  public var keys: Dictionary<Int, Int>.Keys {
    get {
      _indices.keys
    }
  }
  /// Default initializer
  public init() { }
  
  /// The subscript operator, with some indirection
  /// Should be replaced after Dictionary is in
  public subscript(key: Int) -> Tensor<ScalarType> {
    _values[_indices[key]!]
  }
  
  /// L2 norm of the VectorValues
  var norm: Double {
    self._values.map { $0.squared().sum().scalar! }.reduce(0.0, { $0 + $1 })
  }
  
  /// Insert a key value pair
  public mutating func insert(_ key: Int, _ val: Tensor<Self.ScalarType>) {
    assert(_indices[key] == nil)
    
    self._indices[key] = self._values.count
    self._values.append(val)
  }
  
  /// VectorValues + Scalar
  static func + (_ lhs: Self, _ rhs: Self.ScalarType) -> Self {
    var result = lhs
    let _ = result._values.indices.map { result._values[$0] += rhs }
    return result
  }
  
  /// VectorValues + VectorValues
  static func + (_ lhs: Self, _ rhs: Self) -> Self {
    var result = lhs
    for (k, i_r) in rhs._indices {
      if let i_l = lhs._indices[k] {
        result._values[i_l] += rhs._values[i_r]
      } else {
        result.insert(k, rhs._values[i_r])
      }
    }
    return result
  }
  
  /// Scalar * VectorValues
  static func * (_ lhs: Self.ScalarType, _ rhs: Self) -> Self {
    var result = rhs
    let _ = result._values.indices.map { result._values[$0] *= lhs }
    return result
  }
}

extension VectorValues: CustomStringConvertible {
  public var description: String {
    "VectorValues(\n\(_indices.map { "Key: \($0), J: \(_values[$1])\n"}.reduce("", { $0 + $1 }) )"
  }
}

extension VectorValues: Equatable {
  /// Order-aware comparison
  public static func == (lhs: VectorValues, rhs: VectorValues) -> Bool {
    if lhs._indices.keys != rhs._indices.keys {
      return false
    }
    
    for k in lhs._indices.keys {
      if lhs._values[lhs._indices[k]!] != rhs._values[rhs._indices[k]!] {
        return false
      }
    }
    
    return true
  }
}
