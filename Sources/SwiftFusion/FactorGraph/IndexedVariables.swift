/// An assignment of values to an indexed set of variables (e.g. the variables in a `FactorGraph`).
public protocol IndexedValues {
  /// A collection of variable indices.
  associatedtype Indices: Collection where Indices.Collection: Equatable

  /// Identifies a variable.
  typealias Index = Indices.Element

  /// The variables that have assigned values.
  ///
  /// Semantic constraint: The elements are distinct.
  var assignedVariables: Indices { get }
}
