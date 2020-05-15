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

import Foundation
import PenguinGraphs

/// Metropolis algorithm.
extension FactorGraph where
  // The Metropolis algorithm needs to query the graph for neighboring vertices:
  Graph: IncidenceGraph,
  // The Metropolis algorithm needs to see what the variables look like after they have been
  // updated, without necessarily writing to the variables' underlying storage:
  Variables: UpdatedViewable,
  // The Metropolis algorithm needs to know which variables are affected by an update.
  Variables.Update: IndexedVariables, Variables.Index == Variables.Update.Index
{

  /// Returns a draw from a proposal distribution at the given starting state.
  ///
  /// - Returns:
  ///   - update: Describes the change from the starting state to the proposed next state.
  ///   - logQRatio: `log(Q(nextState | startState) / Q(startState | nextState))`, where `Q`
  ///                is the probability density of the proposal distribution.
  public typealias MetropolisProposal<Generator: RandomNumberGenerator> =
    (Variables, inout Generator) -> (update: Variables.Update, logQRatio: Double)

  /// Attempts a Metropolis step on `state` using the given `proposal` distribution, and then
  /// returns and `Update` describing the step to the new state on acceptance, or returns `nil` on
  /// rejection.
  ///
  /// This algorithm exploits the factor graph structure to avoid recomputing factor errors of
  /// unaffected factors.
  ///
  /// - Parameters:
  ///   - applyUpdate: Returns the given `Variables`, with the given `Update` applied.
  public func metropolisStep<Generator: RandomNumberGenerator>(
    startState: Variables,
    proposal: MetropolisProposal<Generator>,
    using generator: inout Generator
  ) -> Variables.Update? {
    // Draw from the proposal distribution.
    let (update, logQRatio) = proposal(startState, &generator)
    let newState = startState.updatedView(update)

    // Only factors neighboring variables that have changed in the new state could have changed.
    let possiblyChangedFactorIds: [Graph.VertexId] = update.indices.flatMap(neighborFactors)

    // Compute the acceptance ratio.
    let logA = possiblyChangedFactorIds.map { possiblyChangedFactorId in
      return error(of: possiblyChangedFactorId, at: startState)
        - error(of: possiblyChangedFactorId, at: newState)
    }.reduce(logQRatio, +)

    // Decide whether we should accept. If not, exit.
    guard logA >= 0 || Double.random(in: 0...1, using: &generator) < exp(logA) else {
      // We reject the update.
      return nil
    }

    // Update the state with the proposal.
    return update
  }
}

/// A type that can be viewed as having an update applied to it.
public protocol UpdatedViewable {
  /// Specifies an update.
  associatedtype Update

  /// Returns a view of `self` with `update` applied.
  func updatedView(_ update: Update) -> Self
}
