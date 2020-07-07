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

/// Protocol for indicating a struct is range-capable
public protocol RangeFunction: Differentiable {
  associatedtype Base: Differentiable
  associatedtype Target: Differentiable
  @differentiable
  static func range(_ from: Base, _ to: Target) -> Double
}

/// Protocol for indicating a struct is bearing-capable
public protocol BearingFunction: Differentiable {
  associatedtype Base: Differentiable
  associatedtype Target: Differentiable
  associatedtype Bearing: Differentiable
  @differentiable
  static func bearing(_ from: Base, _ to: Target) -> Bearing
}

/// Error type for `BearingRangeFactor`.
///
/// This type is composed of two parts:
/// `bearing`: Error in bearing, type `EuclideanVectorN`
/// `range`: Error in range, type `Double`
/// `BE`: The type of the error in bearing
public struct BearingRangeError<BearingError: EuclideanVectorN>:
  AdditiveArithmetic & Differentiable & EuclideanVectorN
{
  @differentiable
  public init(bearing: BearingError, range: Double) {
    self.bearing = bearing
    self.range = range
  }

  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
    lhs.bearing += rhs.bearing
    lhs.range += rhs.range
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
    lhs.bearing -= rhs.bearing
    lhs.range -= rhs.range
  }

  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
    lhs.bearing *= rhs
    lhs.range *= rhs
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    return bearing.dot(other.bearing) + range * other.range
  }

  public init<Source: Collection>(_ scalars: Source) where Source.Element == Double {
    let bearingPrefix = scalars.prefix(BearingError.dimension)
    self.bearing = BearingError(bearingPrefix)
    self.range = scalars[bearingPrefix.endIndex]
  }

  public static var dimension: Int {
    return BearingError.dimension + 1
  }

  public static var standardBasis: [Self] {
    return BearingError.standardBasis.map { BearingRangeError(bearing: $0, range: 0) }
      + [BearingRangeError(bearing: BearingError.zero, range: 1)]
  }

  public static var zero: Self {
    return BearingRangeError(bearing: BearingError.zero, range: 0)
  }

  public var bearing: BearingError
  public var range: Double
}

extension BearingRangeError: Equatable {}

/// A `NonlinearFactor` that calculates the bearing and range error of one pose and one landmark
///
public struct BearingRangeFactor<BearingRangeFunction: BearingFunction & RangeFunction>: NonlinearFactor
where BearingRangeFunction.Base.TangentVector: EuclideanVectorN,
      BearingRangeFunction.Target.TangentVector: EuclideanVectorN,
      BearingRangeFunction.Bearing: EuclideanVectorN
{
  public typealias Base = BearingRangeFunction.Base
  public typealias Target = BearingRangeFunction.Target
  public typealias Bearing = BearingRangeFunction.Bearing
  
  var key1: Int
  var key2: Int
  @noDerivative
  public var keys: Array<Int> {
    get {
      [key1, key2]
    }
  }

  public var bearing: Bearing
  public var range: Double

  public typealias Output = Error

  /// Create a BearingRangeFactor.
  ///
  /// `key1`: Base
  /// `key2`: Target
  /// `bearing`: Measured bearing of `key2` looking from `key1`
  /// `range`: Measured range between `key1` and `key2`
  ///
  public init (_ key1: Int, _ key2: Int, _ bearing: Bearing, _ range: Double) {
    self.key1 = key1
    self.key2 = key2
    self.bearing = bearing
    self.range = range
  }
  typealias ScalarType = Double

  /// Input is a dictionary of `Key` to `Value` pairs, and the output is the scalar
  /// error value
  ///
  /// Interpretation
  /// ================
  /// `Input`: the input values as key-value pairs
  ///
  @differentiable(wrt: values)
  public func error(_ values: Values) -> Double {
    let error_vector = errorVector(values)
    return error_vector.range * error_vector.range + error_vector.bearing.squaredNorm
  }

  @differentiable(wrt: values)
  public func errorVector(_ values: Values) -> BearingRangeError<Bearing> {
    let actual_bearing = BearingRangeFunction.bearing(values[key1, as: Base.self], values[key2, as: Target.self])
    let actual_range = BearingRangeFunction.range(values[key1, as: Base.self], values[key2, as: Target.self])
    let error_range = (actual_range - range)
    let error_bearing = (actual_bearing - bearing)
    
    return BearingRangeError(bearing: error_bearing, range: error_range)
  }

  public func linearize(_ values: Values) -> OldJacobianFactor {
    return OldJacobianFactor(of: self.errorVector, at: values)
  }
}
