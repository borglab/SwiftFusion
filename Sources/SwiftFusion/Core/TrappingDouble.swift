// Copyright 2020 The SwiftFusion Authors. All Rights Reserved.
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

import _Differentiation
#if canImport(PythonKit)
import PythonKit
#endif
// import TensorFlow

/// A wrapper for `Double` that traps instead of allowing `NaN`.
public struct TrappingDouble: AdditiveArithmetic, Differentiable, Equatable {
  public var value: Swift.Double

  public init(_ value: Swift.Double) {
    assert(!value.isNaN)
    self.value = value
  }
}

#if TRAPPING_DOUBLE
/// A wrapper for `Double` that traps instead of allowing `NaN`.
///
/// This typealias makes it easier to use unmodified existing code with `TrappingDouble`.
///
/// If you really need the normal Swift `Double`, use `Swift.Double` to circumvent this typealias.
public typealias Double = TrappingDouble
#endif

// The rest of this file provides conformances and functions that make the interface to
// `TrappingDouble` very close to the interface to `Double`.

extension TrappingDouble: BinaryFloatingPoint {
  public typealias RawSignificand = Swift.Double.RawSignificand
  public typealias RawExponent = Swift.Double.RawExponent

  public static var exponentBitCount: Int { Swift.Double.exponentBitCount }
  public static var significandBitCount: Int { Swift.Double.significandBitCount }
  public var exponentBitPattern: Self.RawExponent { value.exponentBitPattern }
  public var significandBitPattern: Self.RawSignificand { value.significandBitPattern }
  public var binade: Self { Self(value.binade) }
  public var significandWidth: Int { value.significandWidth }
  public typealias Exponent = Swift.Double.Exponent
  public typealias Stride = Swift.Double.Stride
  public typealias Magnitude = Self

  public mutating func formSquareRoot() { value.formSquareRoot() }
  public mutating func addProduct(_ lhs: Self, _ rhs: Self) { value.addProduct(lhs.value, rhs.value) }
  public var nextUp: Self { Self(value.nextUp) }
  public func isEqual(to other: Self) -> Bool { value.isEqual(to: other.value) }
  public func isLess(than other: Self) -> Bool { value.isLess(than: other.value) }
  public func isLessThanOrEqualTo(_ other: Self) -> Bool { value.isLessThanOrEqualTo(other.value) }
  public var isNormal: Bool { value.isNormal }
  public var isFinite: Bool { value.isFinite }
  public var isZero: Bool { value.isZero }
  public var isSubnormal: Bool { value.isSubnormal }
  public var isInfinite: Bool { value.isInfinite }
  public var isNaN: Bool { value.isNaN }
  public var isSignalingNaN: Bool { value.isSignalingNaN }
  public var isCanonical: Bool { value.isCanonical }
  public func distance(to other: Self) -> Self.Stride { value.distance(to: other.value) }
  public func advanced(by n: Swift.Double.Stride) -> Self { Self(value.advanced(by: n)) }
  public var magnitude: Self { Self(value.magnitude) }

  public static var nan: Self { Self(Swift.Double.nan) }
  public static var signalingNaN: Self { Self(Swift.Double.signalingNaN) }
  public static var infinity: Self { Self(Swift.Double.infinity) }
  public static var greatestFiniteMagnitude: Self { Self(Swift.Double.greatestFiniteMagnitude) }
  public var ulp: Self { Self(value.ulp) }
  public static var leastNormalMagnitude: Self { Self(Swift.Double.leastNormalMagnitude) }
  public static var leastNonzeroMagnitude: Self { Self(Swift.Double.leastNonzeroMagnitude) }
  public var sign: FloatingPointSign { value.sign }
  public var exponent: Self.Exponent { value.exponent }
  public var significand: Self { Self(value.significand) }
  public mutating func formRemainder(dividingBy other: Self) { value.formRemainder(dividingBy: other.value) }
  public mutating func formTruncatingRemainder(dividingBy other: Self) { value.formTruncatingRemainder(dividingBy: other.value) }

  public init(sign: FloatingPointSign, exponentBitPattern: Self.RawExponent, significandBitPattern: Self.RawSignificand) {
    self.value = Swift.Double(sign: sign, exponentBitPattern: exponentBitPattern, significandBitPattern: significandBitPattern)
  }
  public init(sign: FloatingPointSign, exponent: Self.Exponent, significand: Self) {
    self.value = Swift.Double(sign: sign, exponent: exponent, significand: significand.value)
  }

  public mutating func round(_ rule: FloatingPointRoundingRule) { value.round(rule) }
}

extension TrappingDouble: TensorFlowFloatingPoint {
  public var xlaScalarWrapper: XLAScalarWrapper { return value.xlaScalarWrapper }
  public static var xlaTensorScalarTypeRawValue: UInt32 { return Swift.Double.xlaTensorScalarTypeRawValue }
  @inlinable
  public static var tensorFlowDataType: TensorDataType { return Swift.Double.tensorFlowDataType }
}

#if canImport(PythonKit)
extension TrappingDouble: PythonConvertible {
  public var pythonObject: PythonObject { value.pythonObject }
}
#endif

extension TrappingDouble: Comparable {
  public static func < (_ lhs: Self, _ rhs: Self) -> Bool {
    return lhs.value < rhs.value
  }
}

extension TrappingDouble: ExpressibleByFloatLiteral {
  public init(floatLiteral: Swift.Double) {
    self.init(floatLiteral)
  }
}

extension TrappingDouble: ExpressibleByIntegerLiteral {
  public init(integerLiteral: Int) {
    self.init(Swift.Double(integerLiteral))
  }
}

extension TrappingDouble: CustomStringConvertible {
  public var description: String {
    return "\(value)"
  }
}

extension TrappingDouble {
  init?<S: StringProtocol>(_ text: S) {
    guard let value = Swift.Double(text) else {
      return nil
    }
    self.init(value)
  }

  @differentiable(reverse)
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    return Self(lhs.value + rhs.value)
  }

  @differentiable(reverse)
  public static func - (_ lhs: Self, _ rhs: Self) -> Self {
    return Self(lhs.value - rhs.value)
  }

  @differentiable(reverse)
  public static func += (_ lhs: inout Self, _ rhs: Self) {
    lhs.value += rhs.value
  }

  @differentiable(reverse)
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
    lhs.value -= rhs.value
  }

  @differentiable(reverse)
  public static prefix func - (_ v: Self) -> Self {
    return Self(-v.value)
  }

  @differentiable(reverse)
  public static func * (_ lhs: Self, _ rhs: Self) -> Self {
    return Self(lhs.value * rhs.value)
  }

  @differentiable(reverse)
  public static func / (_ lhs: Self, _ rhs: Self) -> Self {
    return Self(lhs.value / rhs.value)
  }

  @differentiable(reverse)
  public static func *= (_ lhs: inout Self, _ rhs: Self) {
    lhs.value *= rhs.value
  }

  @differentiable(reverse)
  public static func /= (_ lhs: inout Self, _ rhs: Self) {
    lhs.value /= rhs.value
  }

  public static var ulpOfOne: Self {
    return Self(Swift.Double.ulpOfOne)
  }

  public static var pi: Self {
    return Self(Swift.Double.pi)
  }
}

public func abs(_ x: TrappingDouble) -> TrappingDouble {
  return TrappingDouble(abs(x.value))
}

public func log(_ x: TrappingDouble) -> TrappingDouble {
  return TrappingDouble(log(x.value))
}

@differentiable(reverse)
public func sin(_ x: TrappingDouble) -> TrappingDouble {
  return TrappingDouble(sin(x.value))
}

@differentiable(reverse)
public func cos(_ x: TrappingDouble) -> TrappingDouble {
  return TrappingDouble(cos(x.value))
}

@differentiable(reverse)
public func asin(_ x: TrappingDouble) -> TrappingDouble {
  return TrappingDouble(asin(x.value))
}

@differentiable(reverse)
public func acos(_ x: TrappingDouble) -> TrappingDouble {
  return TrappingDouble(acos(x.value))
}

public func atan2(_ x: TrappingDouble, _ y: TrappingDouble) -> TrappingDouble {
  return TrappingDouble(atan2(x.value, y.value))
}

@differentiable(reverse)
public func tan(_ x: TrappingDouble) -> TrappingDouble {
  return TrappingDouble(tan(x.value))
}

public func sqrt(_ x: TrappingDouble) -> TrappingDouble {
  return TrappingDouble(sqrt(x.value))
}

