/// A vector with a compiletime fixed dimension.
public protocol FixedDimensionVector {
  static var dimension: Int { get }

  subscript(_ index: Int) -> Double { get set }

  static var zero: Self { get }
}

extension FixedDimensionVector {
  public init<C: Collection>(_ scalars: C) where C.Element == Double {
    self = Self.zero
    for (index, scalar) in zip(0..<(Self.dimension), scalars) {
      self[index] = scalar
    }
  }

  public var scalars: LazyMapCollection<Range<Int>, Double> {
    return (0..<(Self.dimension)).lazy.map { self[$0] }
  }

  public static var standardBasis: LazyMapCollection<Range<Int>, Self> {
    return (0..<(Self.dimension)).lazy.map { index in
      var vector = Self.zero
      vector[index] = 1
      return vector
    }
  }
}
