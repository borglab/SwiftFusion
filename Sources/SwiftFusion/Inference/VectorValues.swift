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

/// The class that holds Key-Vector pairs.
public struct VectorValues: KeyPathIterable {
  public typealias ScalarType = Double
  var _values: [Vector] = []
  
  /// Dictionary from Key to index
  @noDerivative var _indices: Dictionary<Int, Int> = [:]
  
  public var keys: Dictionary<Int, Int>.Keys {
    get {
      _indices.keys
    }
  }
  /// Default initializer
  public init() { }
  
  public subscript(key: Int) -> Vector {
    _values[_indices[key]!]
  }

  public subscript(key: Int, default defaultValue: Vector) -> Vector {
    get {
      if let index = _indices[key] {
        return _values[index]
      }
      return defaultValue
    }
    set(newValue) {
      if let index = _indices[key] {
        _values[index] = newValue
        return
      }
      insert(key, newValue)
    }
  }
  
  /// Insert a key value pair
  public mutating func insert(_ key: Int, _ val: Vector) {
    assert(_indices[key] == nil)
    
    self._indices[key] = self._values.count
    self._values.append(val)
  }
  
}

extension VectorValues: EuclideanVectorSpace {

  // NOTE: Most of these are boilerplate that should be synthesized automatically. However, the
  // current synthesis functionality can't deal with the `_indices` property. So we have to
  // implement it manually for now.

  // MARK: - Differentiable conformance.

  public typealias TangentVector = Self

  // MARK: - AdditiveArithmetic conformance.

  public static func += (_ lhs: inout VectorValues, _ rhs: VectorValues) {
    for key in rhs.keys {
      let rhsVector = rhs[key]
      lhs[key, default: Vector(zeros: rhsVector.dimension)] += rhsVector
    }
  }

  public static func + (_ lhs: VectorValues, _ rhs: VectorValues) -> VectorValues {
    var result = lhs
    result += rhs
    return result
  }

  public static func -= (_ lhs: inout VectorValues, _ rhs: VectorValues) {
    for key in rhs.keys {
      let rhsVector = rhs[key]
      lhs[key, default: Vector(zeros: rhsVector.dimension)] -= rhsVector
    }
  }

  public static func - (_ lhs: VectorValues, _ rhs: VectorValues) -> VectorValues {
    var result = lhs
    result -= rhs
    return result
  }

  public static var zero: VectorValues {
    return VectorValues()
  }

  // MARK: - VectorProtocol conformance

  public typealias VectorValuesSpaceScalar = Double

  public mutating func add(_ x: Double) {
    for index in _values.indices {
      _values[index] += x
    }
  }

  public func adding(_ x: Double) -> VectorValues {
    var result = self
    result.add(x)
    return result
  }

  public mutating func subtract(_ x: Double) {
    for index in _values.indices {
      _values[index] -= x
    }
  }

  public func subtracting(_ x: Double) -> VectorValues {
    var result = self
    result.subtract(x)
    return result
  }

  public mutating func scale(by scalar: Double) {
    for index in _values.indices {
      _values[index] *= scalar
    }
  }

  public func scaled(by scalar: Double) -> VectorValues {
    var result = self
    result.scale(by: scalar)
    return result
  }

  // MARK: - Additional EuclideanVectorSpace requirements.

  public var squaredNorm: Double {
    self._values.map { $0.squared().sum() }.reduce(0.0, { $0 + $1 })
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
