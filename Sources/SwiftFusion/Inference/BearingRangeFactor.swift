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
//
///// Protocol for indicating a struct is range-capable
//public protocol HasRange: Differentiable {
//  associatedtype A: Differentiable
//  associatedtype B: Differentiable where B.TangentVector: EuclideanVectorSpace & VectorConvertible & AdditiveArithmetic
//  @differentiable
//  static func range(_ from: A, _ to: B) -> Double
//}
//
//public struct PoseToPointRange2D: HasRange {
//  @differentiable
//  public static func range(_ from: Pose2, _ to: Vector2) -> Double {
//    (to - from.t).norm
//  }
//}
//
//public struct PointToPointRange2D: HasRange {
//  @differentiable
//  public static func range(_ from: Vector2, _ to: Vector2) -> Double {
//    (to - from).norm
//  }
//}
//
///// Protocol for indicating a struct is bearing-capable
//public protocol HasBearing: Differentiable {
//  associatedtype A: LieGroup where A.TangentVector: VectorConvertible
//  associatedtype B: Differentiable where B.TangentVector: EuclideanVectorSpace & VectorConvertible & AdditiveArithmetic
//  associatedtype C: EuclideanVectorSpace & Differentiable & AdditiveArithmetic & VectorProtocol & TangentStandardBasis
//  @differentiable
//  static func bearing(_ from: A, _ to: B) -> C
//}
//
//public struct PoseToPointBearing2D: HasBearing {
//  @differentiable
//  public static func bearing(_ from: Pose2, _ to: Vector2) -> Vector1 {
//    let dx = (to - from.t)
//    return Vector1(between(Rot2(c: dx.x / dx.norm, s: dx.y / dx.norm), from.rot).theta)
//  }
//}
//
//public struct BearingRangeError<BE: TangentStandardBasis, RE: TangentStandardBasis>:
//  Differentiable & KeyPathIterable & TangentStandardBasis & VectorConvertible {
//  @differentiable
//  public init(_ vector: Vector) {
//
//  }
//
//  public var vector: Vector
//
//  public var bearing: BE
//  public var range: RE
//}
//
///// A `NonlinearFactor` that calculates the bearing and range error of one pose and one landmark
/////
///// Input is a dictionary of `Key` to `Value` pairs, and the output is the scalar
///// error value
/////
///// Interpretation
///// ================
///// `Input`: the input values as key-value pairs
/////
//public struct BearingRangeFactor<B: HasBearing, R: HasRange>: NonlinearFactor
//where B.A.TangentVector == B.A.Coordinate.LocalCoordinate,
//      B.A == R.A
//{
//
//  var key1: Int
//  var key2: Int
//  @noDerivative
//  public var keys: Array<Int> {
//    get {
//      [key1, key2]
//    }
//  }
//
//  public var bearing: B.C
//  public var range: Double
//
//  public typealias Output = Error
//
//  public init (_ key1: Int, _ key2: Int, _ bearing: B.C, _ range: Double) {
//    self.key1 = key1
//    self.key2 = key2
//    self.bearing = bearing
//    self.range = range
//  }
//  typealias ScalarType = Double
//
//  /// Returns the `error` of the factor.
//  @differentiable(wrt: values)
//  public func error(_ values: Values) -> Double {
//    let actual_bearing = B.bearing(values[key1, as: B.A.self], values[key2, as: B.B.self])
//    let actual_range = R.range(values[key1, as: B.A.self], values[key2, as: R.B.self])
//    let error_range = (actual_range - range) * (actual_range - range)
//    let error_bearing = (actual_bearing - bearing).squaredNorm
//    return error_range + error_bearing
//  }
//
//  @differentiable(wrt: values)
//  public func errorVector(_ values: Values) -> BearingRangeError<B.C, Double> {
//    let actual_bearing = B.bearing(values[key1, as: B.A.self], values[key2, as: B.B.self])
//    let actual_range = R.range(values[key1, as: B.A.self], values[key2, as: R.B.self])
//
//    return BearingRangeError(bearing: actual_bearing - bearing, range: actual_range - range)
//  }
//
//  public func linearize(_ values: Values) -> JacobianFactor {
//    return JacobianFactor(of: self.errorVector, at: values)
//  }
//}


/// A `NonlinearFactor` that calculates the bearing and range error of one pose and one landmark
///
/// Input is a dictionary of `Key` to `Value` pairs, and the output is the scalar
/// error value
///
/// Interpretation
/// ================
/// `Input`: the input values as key-value pairs
///
public struct BearingRangeFactor2D: NonlinearFactor
{
  
  var key1: Int
  var key2: Int
  @noDerivative
  public var keys: Array<Int> {
    get {
      [key1, key2]
    }
  }
  
  public var bearing: Double
  public var range: Double
  
  public typealias Output = Error
  
  public init (_ key1: Int, _ key2: Int, _ bearing: Double, _ range: Double) {
    self.key1 = key1
    self.key2 = key2
    self.bearing = bearing
    self.range = range
  }
  typealias ScalarType = Double
  
  /// Returns the `error` of the factor.
  @differentiable(wrt: values)
  public func error(_ values: Values) -> Double {
    let from = values[key1, as: Pose2.self]
    let to = values[key2, as: Vector2.self]
    let dx = (to - from.t)
    
    let actual_bearing = between(Rot2(c: dx.x / dx.norm, s: dx.y / dx.norm), from.rot).theta
    let actual_range = dx.squaredNorm
    let error_range = (actual_range - range) * (actual_range - range)
    let error_bearing = (actual_bearing - bearing)*(actual_bearing - bearing)
    return error_range + error_bearing
  }
  
  @differentiable(wrt: values)
  public func errorVector(_ values: Values) -> Vector2 {
    let from = values[key1, as: Pose2.self]
    let to = values[key2, as: Vector2.self]
    let dx = (to - from.t)
    
    let actual_bearing = between(Rot2(c: dx.x / dx.norm, s: dx.y / dx.norm), from.rot).theta
    let actual_range = dx.squaredNorm
    let error_range = (actual_range - range) * (actual_range - range)
    let error_bearing = (actual_bearing - bearing)*(actual_bearing - bearing)
    
    return Vector2(error_bearing, error_range)
  }
  
  public func linearize(_ values: Values) -> JacobianFactor {
    return JacobianFactor(of: self.errorVector, at: values)
  }
}
