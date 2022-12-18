import ShapedArray

extension ShapedArray {
  @differentiable(reverse)
  public static func * (_ lhs: Self, _ rhs: Self) -> Self {
    var result = lhs
    result *= rhs
    return result
  }

  @differentiable(reverse)
  public static func *= (_ lhs: inout Self, _ rhs: Self) {
    if lhs._isScalarZero {
      lhs = rhs
    } else if !rhs._isScalarZero {
      for i in 0..<lhs.buffer.count {
        lhs.buffer[i] *= rhs.buffer[i]
      }
    }
  }

  @derivative(of:*=)
  @usableFromInline
  static func vjpMultiplyEquals(_ lhs: inout Self, _ rhs: Self) -> (
    value: (), pullback: (inout Self) -> Self
  ) {
    lhs *= rhs
    func pullback(_ v: inout Self) -> Self {
      return v
    }
    return ((), pullback)
  }

  /// squared
  @differentiable(reverse)
  @usableFromInline
  func squared() -> Self {
    self * self
  }
}
