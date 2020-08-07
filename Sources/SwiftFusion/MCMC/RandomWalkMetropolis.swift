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
import TensorFlow

/// Runs one step of the RWM algorithm with symmetric proposal.
/// Inspired by tfp [RandomWalkMetropolis](https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/RandomWalkMetropolis)
public class RandomWalkMetropolis<State> : TransitionKernel {
  public typealias Results = Double // target_log_prob_fn of previously accepted state
  
  let target_log_prob_fn: (State) -> Double
  let new_state_fn : (State)->State
  var sourceOfEntropy: AnyRandomNumberGenerator
  
  public init(
    sourceOfEntropy: RandomNumberGenerator = SystemRandomNumberGenerator(),
    target_log_prob_fn: @escaping (State) -> Double,
    new_state_fn : @escaping (State)->State
  ) {
    self.target_log_prob_fn = target_log_prob_fn
    self.new_state_fn = new_state_fn
    self.sourceOfEntropy = .init(sourceOfEntropy)
  }
  
  /// Runs one iteration of Random Walk Metropolis.
  /// TODO(frank): should this be done with inout params in Value-semantics world?
  public func one_step(_ current_state: State, _ previous_kernel_results: Results) -> (State, Results) {
    
    // calculate next state, and new log probability
    let new_state = new_state_fn(current_state)
    let new_log_prob = target_log_prob_fn(new_state)
    
    // Calculate log of acceptance ratio p(x')/p(x) = log p(x') - log p(x)
    let current_log_prob = previous_kernel_results
    let log_accept_ratio = new_log_prob - current_log_prob
    
    // If p(x')/p(x) >= 1 , i.e., log_accept_ratio >= 0, we always accept
    // otherwise we accept randomly with probability p(x')/p(x).
    // We do this by randomly sampling u from [0,1], and comparing log(u) with log_accept_ratio.
    let u = Double.random(in: 0..<1, using: &sourceOfEntropy)
    if (log(u) <= log_accept_ratio) {
      return (new_state, new_log_prob)
    } else {
      return (current_state, current_log_prob)
    }
  }
  
  /// Initializes side information
  public func bootstrap_results(_ init_state: State) -> Results {
    return target_log_prob_fn(init_state)
  }
}

