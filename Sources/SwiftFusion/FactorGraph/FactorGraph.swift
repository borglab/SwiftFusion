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

import PenguinGraphs

/// A graph where vertices are parititioned into "left" and "right" sets, and where edges are
/// always incident to one "left" vertex and one "right" vertex.
public protocol BipartiteGraph: GraphProtocol
  where VertexId == BipartiteGraphVertexId<LeftVertexId, RightVertexId>
{
  /// Identifies a vertex in the "left" half of the partition.
  associatedtype LeftVertexId

  /// Identifies a vertex in the "right" half of the partition.
  associatedtype RightVertexId
}

/// Identifies an arbitrary vertex in a `BipartiteGraph`.
public enum BipartiteGraphVertexId<LeftVertexId: Equatable, RightVertexId: Equatable>: Equatable {
  case left(LeftVertexId)
  case right(RightVertexId)
}

/// A bipartite graph of "factor vertices" and "variable vertices", where each factor vertex is an
/// error function over the variables that it is connected to.
public protocol FactorGraph: BipartiteGraph {

  // MARK: - Graph.

  /// Identifies a factor.
  typealias FactorId = LeftVertexId

  /// Identifies a variable.
  typealias VariableId = RightVertexId

  // MARK: - Variables.

  /// An assignment of values to variables.
  associatedtype Values: IndexedValues where Values.Index == VariableId

  // MARK: - Factor error function.

  /// Returns the error of the factor `factor`, when the variable values are `values`.
  ///
  /// This can be interpreted as an unnormalized negative log probability.
  ///
  /// Semantic constraint: Given a specific `factor`, this is only allowed to depend on the
  /// values of the variables that are adjacent to `factor`.
  func error(of factor: FactorId, at values: Values) -> Double
}
