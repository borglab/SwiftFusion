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
import PenguinStructures

extension ArrayBuffer: Vector where Element: Vector {
  public struct Scalars: MutableCollection {
    var base: ArrayBuffer

    public typealias Index = NestedIndex<ArrayBuffer.Index, Element.Scalars.Index>
    
    public subscript(i: Index) -> Double {
      get { base[i.outer].scalars[i.inner!] }
      _modify { yield &base[i.outer].scalars[i.inner!] }
    }
    public var startIndex: Index {
      .init(firstValidIn: base, innerCollection: \.scalars)
    }
    public var endIndex: Index {
      .init(endIn: base)
    }
    public func index(after i: Index) -> Index {
      .init(nextAfter: i, in: base, innerCollection: \.scalars)
    }
  }
  
  public var scalars: Scalars {
    get {
      Scalars(base: self)
    }
    _modify {
      var io = Scalars(base: self)
      self = ArrayBuffer() // Avoid CoWs
      yield &io
      swap(&self, &io.base)
    }
  }
  
  public var dimension: Int { self.lazy.map(\.dimension).reduce(0, +) }
  
  /// Replaces lhs with the product of `lhs` and `rhs`.
  @differentiable
  public static func *= (lhs: inout ArrayBuffer, rhs: Double) -> Void {
    if lhs.isEmpty { return }
    if rhs == 0 { lhs = .zero }
    else {
      lhs.update(elementwise: rhs, *=, { l, r in r * l })
    }
  }

  @derivative(of: *=)
  @usableFromInline
  static func vjp_timesEqual(lhs: inout ArrayBuffer, rhs: Double)
    -> (value: Void, pullback: (inout TangentVector)->Double)
  {
    let oldLHS = lhs
    lhs *= rhs
    return ((),
            { tangent in
              tangent *= rhs
              return oldLHS.dot(tangent)
            })
  }

  /// Returns the product of `lhs` and `rhs`.
  @differentiable
  public static func * (lhs: Double, rhs: ArrayBuffer) -> ArrayBuffer {
    if rhs.isEmpty { return rhs }
    if lhs == 0 { return .zero }
    return .init(rhs.lazy.map { lhs * $0 })
  }

  @derivative(of: *)
  @usableFromInline
  static func vjp_timesEqual(lhs: Double, rhs: ArrayBuffer)
    -> (value: ArrayBuffer, pullback: (TangentVector)->(Double, TangentVector))
  {
    return (lhs * rhs, { tangent in (rhs.dot(tangent), lhs * tangent) })
  }

  /// Returns the dot product of `self` with `other`.
  ///
  /// - Requires: `self.tensorShapeIsCompatible(withThatOf: other)`
  @differentiable
  public func dot(_ other: ArrayBuffer) -> Double {
    assert(self.tensorShapeIsCompatible(withThatOf: other))
    if self.isEmpty || other.isEmpty { return 0 }
    return self.withUnsafeBufferPointer { lhs in
      other.withUnsafeBufferPointer { rhs in
        (0..<lhs.count).reduce(0) { sum, i in sum + lhs[i].dot(rhs[i]) }
      }
    }
  }

  @derivative(of: dot)
  @usableFromInline
  func vjp_dot(_ other: ArrayBuffer)
    -> (value: Double, pullback: (Double)->(TangentVector, TangentVector))
  {
    return (self.dot(other), { r in (r * other, r * self) })
  }

  /// Returns the sum of `lhs` and `rhs`.
  ///
  /// - Requires: `lhs.tensorShapeIsCompatible(withThatOf: rhs)`
  @differentiable
  public static func + (lhs: ArrayBuffer, rhs: ArrayBuffer) -> ArrayBuffer {
    plus(lhs, rhs)
  }
  
  @differentiable
  private static func plus(_ lhs: ArrayBuffer, _ rhs: ArrayBuffer) -> ArrayBuffer {
    if lhs.isEmpty { return rhs }
    if rhs.isEmpty { return lhs }
    return .init(elementwise: lhs, rhs, +)
  }

  @derivative(of: plus)
  private static func vjp_plus(lhs: ArrayBuffer, rhs: ArrayBuffer) 
    -> (value: ArrayBuffer, pullback: (TangentVector)->(TangentVector, TangentVector))
  {
    (lhs + rhs, { x in (x, x) })
  }
  
  /// Returns the result of subtracting `rhs` from `lhs`.
  ///
  /// - Requires: `lhs.tensorShapeIsCompatible(withThatOf: rhs)`
  @differentiable
  public static func - (lhs: ArrayBuffer, rhs: ArrayBuffer) -> ArrayBuffer {
    minus(lhs, rhs)
  }

  /// Returns the result of subtracting `rhs` from `lhs`.
  ///
  /// - Requires: `lhs.tensorShapeIsCompatible(withThatOf: rhs)`
  @differentiable
  private static func minus (_ lhs: ArrayBuffer, _ rhs: ArrayBuffer) -> ArrayBuffer {
    if rhs.isEmpty { return lhs }
    return .init(elementwise: lhs, rhs, -)
  }

  @derivative(of: minus)
  private static func vjp_minus(lhs: ArrayBuffer, rhs: ArrayBuffer) 
    -> (value: ArrayBuffer, pullback: (TangentVector)->(TangentVector, TangentVector))
  {
    (lhs - rhs, { x in (x, .zero - x) })
  }
  
  /// Replaces `lhs` with the sum of `lhs` and `rhs`
  ///
  /// - Requires: `lhs.tensorShapeIsCompatible(withThatOf: rhs)`
  @differentiable
  public static func += (lhs: inout ArrayBuffer, rhs: ArrayBuffer) {
    plusEquals(&lhs, rhs)
  }

  /// Replaces `lhs` with the sum of `lhs` and `rhs`
  ///
  /// - Requires: `lhs.tensorShapeIsCompatible(withThatOf: rhs)`
  @differentiable
  private static func plusEquals (_ lhs: inout ArrayBuffer, _ rhs: ArrayBuffer) {
    if rhs.isEmpty { return }
    else if lhs.isEmpty { lhs = rhs }
    else { lhs.update(elementwiseWith: rhs, +=, +) }
  }

  @derivative(of: plusEquals)
  private static func vjp_plusEquals(lhs: inout ArrayBuffer, rhs: ArrayBuffer) 
    -> (value: Void, pullback: (inout TangentVector)->(TangentVector))
  {
    lhs += rhs
    return ((), { x in x })
  }
  
  /// Returns the result of subtracting `rhs` from `lhs`.
  ///
  /// - Requires: `lhs.tensorShapeIsCompatible(withThatOf: rhs)`
  @differentiable
  public static func -= (lhs: inout ArrayBuffer, rhs: ArrayBuffer) {
    minusEquals(&lhs, rhs)
  }

  /// Returns the result of subtracting `rhs` from `lhs`.
  ///
  /// - Requires: `lhs.tensorShapeIsCompatible(withThatOf: rhs)`
  @differentiable
  private static func minusEquals(_ lhs: inout ArrayBuffer, _ rhs: ArrayBuffer) {
    if rhs.isEmpty { return }
    else if lhs.isEmpty { lhs = lhs - rhs }
    else { lhs.update(elementwiseWith: rhs, -=, -) }
  }

  @derivative(of: minusEquals)
  private static func vjp_minusEquals(lhs: inout ArrayBuffer, rhs: ArrayBuffer) 
    -> (value: Void, pullback: (inout TangentVector)->(TangentVector))
  {
    lhs -= rhs
    return ((), { x in .zero - x })
  }
}
