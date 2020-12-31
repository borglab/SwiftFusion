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
import TensorFlow

/// A view of a `Tensor` as a `Vector`.
public struct TensorVector {
  /// A type that can represent all of this vector's scalar values in a standard basis.
  public struct Scalars
    : RandomAccessCollection, MutableCollection, Differentiable, AdditiveArithmetic
  {
    public typealias TangentVector = Self

    fileprivate var storage: Tensor<Double>

    /// The position of the first element, or `endIndex` if `self.isEmpty`.
    public var startIndex: Int { 0 }

    /// The position one step beyond the last contained element.
    public var endIndex: Int { storage.scalarCount }

    /// Accesses the scalar at `i`.
    public subscript(i: Int) -> Double {
      get { storage[i].scalarized() }
      set { storage[i] = Tensor(newValue) }
    }
  }

  /// This vector's scalar values in a standard basis.
  public var scalars: Scalars

  /// The shape of this vector's underlying tensor.
  @noDerivative public private(set) var shape: TensorShape

  /// This vector's underlying tensor.
  @differentiable
  public var tensor: Tensor<Double> {
    get {
      return scalars.storage.reshaped(to: shape)
    }
    set {
      self.scalars = Scalars(storage: newValue.reshaped(to: [tensor.scalarCount]))
      self.shape = newValue.shape
    }
  }

  @derivative(of: tensor)
  @usableFromInline
  func vjpTensor() -> (value: Tensor<Double>, pullback: (Tensor<Double>) -> Self) {
    (tensor, { Self($0) })
  }

  /// Creates an instance that views `tensor` as a `Vector`.
  @differentiable
  public init(_ tensor: Tensor<Double>) {
    self.scalars = Scalars(storage: tensor.reshaped(to: [tensor.scalarCount]))
    self.shape = tensor.shape
  }
}

extension TensorVector: AdditiveArithmetic {
  @differentiable
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    Self(lhs.tensor + rhs.tensor)
  }

  @differentiable
  public static func - (_ lhs: Self, _ rhs: Self) -> Self {
    Self(lhs.tensor - rhs.tensor)
  }

  @differentiable
  public static func += (_ lhs: inout Self, _ rhs: Self) {
    lhs = Self(lhs.tensor + rhs.tensor)
  }

  @differentiable
  public static func -= (_ lhs: inout Self, _ rhs: Self) {
    lhs = Self(lhs.tensor - rhs.tensor)
  }

  public static var zero: Self {
    Self(Tensor.zero)
  }
}

extension TensorVector: Differentiable {
  public typealias TangentVector = Self

  public mutating func move(along direction: TangentVector) {
    self.tensor += direction.tensor
  }
}

extension TensorVector: Vector {
  @differentiable
  public static func *= (_ lhs: inout Self, _ rhs: Double) {
    lhs = Self(lhs.tensor * rhs)
  }

  @differentiable
  public func dot(_ other: Self) -> Double {
    (self.tensor * other.tensor).sum().scalarized()
  }

  public var dimension: Int {
    self.scalars.storage.scalarCount
  }

  /// Returns the result of calling `body` on the scalars of `self`.
  public func withUnsafeBufferPointer<R>(
    _ body: (UnsafeBufferPointer<Double>) throws -> R
  ) rethrows -> R {
    try self.tensor.scalars.withUnsafeBufferPointer(body)
  }

  /// Returns the result of calling `body` on the scalars of `self`.
  public mutating func withUnsafeMutableBufferPointer<R>(
    _ body: (inout UnsafeMutableBufferPointer<Double>) throws -> R
  ) rethrows -> R {
    var scalars = self.tensor.scalars
    let r = try scalars.withUnsafeMutableBufferPointer { b in
      try body(&b)
    }
    self.tensor = Tensor(shape: self.shape, scalars: scalars)
    return r
  }
}
