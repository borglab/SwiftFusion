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

public struct PointToPointRange2D: RangeFunction {
  @differentiable
  public static func range(_ from: Vector2, _ to: Vector2) -> Double {
    (to - from).norm
  }
}

/// Protocol for indicating a struct is bearing-capable
public protocol BearingFunction: Differentiable {
  associatedtype Base: Differentiable
  associatedtype Target: Differentiable
  associatedtype Bearing: Differentiable
  @differentiable
  static func bearing(_ from: Base, _ to: Target) -> Bearing
}

public struct PoseToPointBearingRange2D: RangeFunction & BearingFunction {
  @differentiable
  public static func bearing(_ from: Pose2, _ to: Vector2) -> Vector1 {
    let dx = (to - from.t)
    return Vector1(between(Rot2(c: dx.x / dx.norm, s: dx.y / dx.norm), from.rot).theta)
  }
  
  @differentiable
  public static func range(_ from: Pose2, _ to: Vector2) -> Double {
    (to - from.t).norm
  }
}

public struct BearingRangeError<BE: TangentStandardBasis & VectorConvertible & FixedDimensionVector>:
  Differentiable & KeyPathIterable & TangentStandardBasis & VectorConvertible {
  @differentiable
  public init(_ vector: Vector) {
    precondition(vector.dimension == BE.dimension + 1)
    let part = vector.scalars.differentiablePartition(BE.dimension)
    bearing = BE(Vector(part.a))
    range = vector.scalars[BE.dimension]
  }
  
  @differentiable
  public init(bearing: BE, range: Double) {
    self.bearing = bearing
    self.range = range
  }

  @differentiable
  public var vector: Vector {
    get {
      Vector(bearing.vector.scalars + [range])
    }
  }

  public var bearing: BE
  public var range: Double
}

/// A `NonlinearFactor` that calculates the bearing and range error of one pose and one landmark
///
/// Input is a dictionary of `Key` to `Value` pairs, and the output is the scalar
/// error value
///
/// Interpretation
/// ================
/// `Input`: the input values as key-value pairs
///
public struct BearingRangeFactor<BearingRangeFunction: BearingFunction & RangeFunction>: NonlinearFactor
where BearingRangeFunction.Base.TangentVector: VectorConvertible,
      BearingRangeFunction.Target.TangentVector: VectorConvertible,
      BearingRangeFunction.Bearing: Differentiable & TangentStandardBasis & EuclideanVector & VectorConvertible & FixedDimensionVector
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

  public init (_ key1: Int, _ key2: Int, _ bearing: Bearing, _ range: Double) {
    self.key1 = key1
    self.key2 = key2
    self.bearing = bearing
    self.range = range
  }
  typealias ScalarType = Double

  /// Returns the `error` of the factor.
  @differentiable(wrt: values)
  public func error(_ values: Values) -> Double {
    let actual_bearing = BearingRangeFunction.bearing(values[key1, as: Base.self], values[key2, as: Target.self])
    let actual_range = BearingRangeFunction.range(values[key1, as: Base.self], values[key2, as: Target.self])
    let error_range = (actual_range - range) * (actual_range - range)
    let error_bearing = (actual_bearing.vector - bearing.vector).squaredNorm
    return error_range + error_bearing
  }

  @differentiable(wrt: values)
  public func errorVector(_ values: Values) -> BearingRangeError<Bearing> {
    let actual_bearing = BearingRangeFunction.bearing(values[key1, as: Base.self], values[key2, as: Target.self])
    let actual_range = BearingRangeFunction.range(values[key1, as: Base.self], values[key2, as: Target.self])
    let error_range = (actual_range - range)
    let error_bearing = (actual_bearing - bearing)
    
    return BearingRangeError(bearing: error_bearing, range: error_range)
  }

  public func linearize(_ values: Values) -> JacobianFactor {
    return JacobianFactor(of: self.errorVector, at: values)
  }
}
