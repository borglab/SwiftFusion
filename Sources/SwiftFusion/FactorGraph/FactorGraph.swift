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

/// A bipartite graph of "factor vertices" and "variable vertices", where each factor vertex is an
/// error function over the variables that it is connected to.
public protocol FactorGraph {

  // MARK: - Graph.

  /// The type of the graph.
  associatedtype Graph: GraphProtocol

  /// Bipartite graph of "factor vertices" and "variable vertices".
  var graph: Graph { get }

  // MARK: - Variables.

  /// The type representing the values of the variables.
  associatedtype Variables: IndexedVariables

  /// Returns `variable`'s vertex id in the graph.
  ///
  /// Note: This is a workaround for the fact that there is nothing like a BipartiteGraph protocol.
  /// If there were, we could simply set Varaibles.Index == Graph.RightVertexId and not have to
  /// convert between them.
  func vertexId(of variable: Variables.Index) -> Graph.VertexId

  /// Returns the variable index for the variable vertex `vertex`.
  ///
  /// Precondition: `vertex` refers to a variable vertex (not a facor vertex).
  ///
  /// Note: This is a workaround for the fact that there is nothing like a BipartiteGraph protocol.
  /// If there were, we could simply set Varaibles.Index == Graph.RightVertexId and not have to
  /// convert between them.
  func variableIndex(of vertex: Graph.VertexId) -> Variables.Index

  // MARK: - Factor error function.

  /// Returns the error of the factor `factor`, at `point`.
  ///
  /// This can be interpreted as an unnormalized negative log probability.
  ///
  /// Precondition: `factor` refers to a factor vertex.
  func error(of factor: Graph.VertexId, at point: Variables) -> Double
}

/// Graph traversal helpers.
extension FactorGraph where Graph: IncidenceGraph {
  /// Returns the vertex ids of the factors neighboring `variableIndex`.
  public func neighborFactors(of variableIndex: Variables.Index) -> [Graph.VertexId] {
    let variableVertexId = self.vertexId(of: variableIndex)
    return graph.edges(from: variableVertexId).map { graph.destination(of: $0) }
  }

  /// Returns the variable ids of the variables neighboring `factor`.
  public func neighborVariables(of factor: Graph.VertexId) -> [Variables.Index] {
    return graph.edges(from: factor).map { self.variableIndex(of: graph.destination(of: $0)) }
  }
}
