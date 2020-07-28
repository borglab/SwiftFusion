import TensorFlow

/// A vector in a Euclidean vector space whose dimension is fixed at compile time.
public protocol EuclideanVectorN: EuclideanVector {
  /// The dimension of the vector.
  static var dimension: Int { get }

  /// A standard basis of vectors.
  static var standardBasis: [Self] { get }

  /// Returns the result of calling `body` on the scalars of `self`.
  ///
  /// A default is provided that returns a pointer to `self`.
  func withUnsafeBufferPointer<R>(
    _ body: (UnsafeBufferPointer<Double>) throws -> R
  ) rethrows -> R

  /// Returns the result of calling `body` on the scalars of `self`.
  ///
  /// A default is provided that returns a pointer to `self`.
  mutating func withUnsafeMutableBufferPointer<R>(
    _ body: (UnsafeMutableBufferPointer<Double>) throws -> R
  ) rethrows -> R
}

extension EuclideanVectorN {
  /// Returns the result of calling `body` on the scalars of `self`.
  public func withUnsafeBufferPointer<R>(
    _ body: (UnsafeBufferPointer<Double>) throws -> R
  ) rethrows -> R {
    return try withUnsafePointer(to: self) { p in
      try body(
          UnsafeBufferPointer<Double>(
              start: UnsafeRawPointer(p)
                  .assumingMemoryBound(to: Double.self),
              count: Self.dimension))
    }
  }

  /// Returns the result of calling `body` on the scalars of `self`.
  public mutating func withUnsafeMutableBufferPointer<R>(
    _ body: (UnsafeMutableBufferPointer<Double>) throws -> R
  ) rethrows -> R {
    return try withUnsafeMutablePointer(to: &self) { p in
      try body(
          UnsafeMutableBufferPointer<Double>(
              start: UnsafeMutableRawPointer(p)
                  .assumingMemoryBound(to: Double.self),
              count: Self.dimension))
    }
  }

  /// Accesses the scalar at `i`.
  subscript(i: Int) -> Double {
    _read {
      boundsCheck(i)
      yield withUnsafeBufferPointer { $0.baseAddress.unsafelyUnwrapped[i] }
    }
    _modify {
      boundsCheck(i)
      defer { _fixLifetime(self) }
      yield &withUnsafeMutableBufferPointer { $0.baseAddress }.unsafelyUnwrapped[i]
    }
  }

  /// Traps with a suitable error message if `i` is not the position of an
  /// element in `self`.
  private func boundsCheck(_ i: Int) {
    precondition(i >= 0 && i < Self.dimension, "index out of range")
  }
}

extension EuclideanVectorN {
  /// Creates a vector with the same scalars as `v`.
  ///
  /// - Requires: `Self.dimension == V.dimension`.
  @differentiable
  public init<V: EuclideanVectorN>(_ v: V) {
    precondition(Self.dimension == V.dimension)
    self = Self.zero
    self.withUnsafeMutableBufferPointer { rBuf in
      v.withUnsafeBufferPointer { vBuf in
        for (i, s) in vBuf.enumerated() {
          rBuf[i] = s
        }
      }
    }
  }

  @derivative(of: init(_:))
  @usableFromInline
  static func vjpInit<V: EuclideanVectorN>(_ v: V) -> (value: Self, pullback: (Self) -> V) {
    return (
      Self(v),
      { t in V(t) }
    )
  }

  /// Creates a vector with the scalars from `v1`, followed by the scalars from `v2`.
  ///
  /// - Requires: `Self.dimension == V1.dimension + V2.dimension`.
  @differentiable
  public init<V1: EuclideanVectorN, V2: EuclideanVectorN>(concatenating v1: V1, _ v2: V2) {
    precondition(Self.dimension == V1.dimension + V2.dimension)
    self = Self.zero
    self.withUnsafeMutableBufferPointer { rBuf in
      v1.withUnsafeBufferPointer { v1Buf in
        for (i, s) in v1Buf.enumerated() {
          rBuf[i] = s
        }
      }
      v2.withUnsafeBufferPointer { v2Buf in
        for (i, s) in v2Buf.enumerated() {
          rBuf[i + V1.dimension] = s
        }
      }
    }
  }

  @derivative(of: init(concatenating:_:))
  @usableFromInline
  static func vjpInit<V1: EuclideanVectorN, V2: EuclideanVectorN>(concatenating v1: V1, _ v2: V2) -> (
    value: Self,
    pullback: (Self) -> (V1, V2)
  ) {
    return (
      Self(concatenating: v1, v2),
      { t in
        t.withUnsafeBufferPointer { tBuf in
          var t1 = V1.zero
          t1.withUnsafeMutableBufferPointer { t1Buf in
            for i in t1Buf.indices {
              t1Buf[i] = tBuf[i]
            }
          }
          var t2 = V2.zero
          t2.withUnsafeMutableBufferPointer { t2Buf in
            for i in t2Buf.indices {
              t2Buf[i] = tBuf[i + V1.dimension]
            }
          }
          return (t1, t2)
        }
      }
    )
  }
}

extension EuclideanVectorN {
  /// Creates an instance with the same scalars as `flatTensor`.
  ///
  /// - Reqiures: `flatTensor.shape == [Self.dimension]`.
  @differentiable
  public init(flatTensor: Tensor<Double>) {
    precondition(flatTensor.shape == [Self.dimension])
    self.init(flatTensor.scalars)
  }

  @derivative(of: init(flatTensor:))
  @usableFromInline
  static func vjpInit(flatTensor: Tensor<Double>) -> (
    value: Self,
    pullback: (Self) -> Tensor<Double>
  ) {
    return (Self(flatTensor: flatTensor), { $0.flatTensor })
  }

  /// Returns a `Tensor` with shape `[Self.dimension]` with the same scalars as `self`.
  @differentiable
  public var flatTensor: Tensor<Double> {
    withUnsafeBufferPointer { b in
      return Tensor<Double>(shape: [b.count], scalars: b)
    }
  }

  @derivative(of: flatTensor)
  @usableFromInline
  func vjpFlatTensor() -> (value: Tensor<Double>, pullback: (Tensor<Double>) -> Self) {
    return (self.flatTensor, { Self(flatTensor: $0) })
  }
}
