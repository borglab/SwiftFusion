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

  /// Returns the number of variables.
  public var count: Int {
    return _values.count
  }

  /// MARK: - Differentiable conformance and related properties and helpers.

  /// The product space of the tangent spaces of all the values.
  public typealias TangentVector = VectorValues

  /// `makeTangentVector[i]` produces a type-erased tangent vector for `values[i]`.
  private var makeTangentVector: [(Vector) -> AnyDerivative] = []

  public mutating func move(along direction: VectorValues) {
    for key in direction.keys {
      let index = self._indices[key]!
      self._values[index].move(along: makeTangentVector[index](direction[key]))
    }
  }
  
  /// MARK: - Value manipulation methods.

  /// Access the value at `key`, with type `type`.
  ///
  /// Precondition: The value actually has type `type`.
  @differentiable
  public subscript<T: Differentiable>(key: Int, as type: T.Type) -> T
    where T.TangentVector: VectorConvertible
  {
    get {
      return _values[_indices[key]!].baseAs(type)
    }
    set(newValue) {
      _values[_indices[key]!] = AnyDifferentiable(newValue)
    }
  }

  @derivative(of: subscript)
  @usableFromInline
  func vjpSubscript<T: Differentiable>(key: Int, as type: T.Type)
    -> (value: T, pullback: (T.TangentVector) -> VectorValues)
    where T.TangentVector: VectorConvertible
  {
    return (
      self._values[self._indices[key]!].baseAs(type),
      { (t: T.TangentVector) in
        var vectorValues = VectorValues()
        vectorValues.insert(key, t.vector)
        return vectorValues
      }
    )
  }

  /// Insert a key value pair
  public mutating func insert<T: Differentiable>(_ key: Int, _ val: T)
    where T.TangentVector: VectorConvertible
  {
    assert(_indices[key] == nil)
    
    self._indices[key] = self._values.count
    self._values.append(AnyDifferentiable(val))
    self.makeTangentVector.append({ AnyDerivative(T.TangentVector($0)) })
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
