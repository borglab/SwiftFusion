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

/// The class that holds Key-Value pairs.
public struct Values: Differentiable & KeyPathIterable {
  public typealias ScalarType = Double
  var _values: [AnyDifferentiable] = []
  
  /// Dictionary from Key to index
  @noDerivative
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
  @differentiable
  public subscript(key: Int) -> AnyDifferentiable {
    get {
      _values[_indices[key]!]
    }
    set(newVal) {
      _values[_indices[key]!] = newVal
    }
  }
  
  /// Insert a key value pair
  public mutating func insert(_ key: Int, _ val: AnyDifferentiable) {
    assert(_indices[key] == nil)
    
    self._indices[key] = self._values.count
    self._values.append(val)
  }
  
}

extension Values: CustomStringConvertible {
  public var description: String {
    "Values(\n\(_indices.map { "Key: \($0), J: \(_values[$1])\n"}.reduce("", { $0 + $1 }) )"
  }
}

//extension Values: Equatable {
//  /// Order-aware comparison
//  public static func == (lhs: Values, rhs: Values) -> Bool {
//    if lhs._indices.keys != rhs._indices.keys {
//      return false
//    }
//
//    for k in lhs._indices.keys {
//      if lhs._values[lhs._indices[k]!] != rhs._values[rhs._indices[k]!] {
//        return false
//      }
//    }
//
//    return true
//  }
//}
