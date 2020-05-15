/// A value that stores variables that are indexed by some index type.
public protocol IndexedVariables {
  /// Collection of indices.
  associatedtype Indices: Collection

  /// Single index.
  typealias Index = Indices.Element

  /// The indices of the variables stored in `self`.
  var indices: Indices { get }
}
