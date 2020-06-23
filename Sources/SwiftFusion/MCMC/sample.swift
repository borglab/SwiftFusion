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

/// Implements Markov chain Monte Carlo via repeated TransitionKernel steps
/// Inspired by tfp [sample_chain](https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/sample_chain)
public func sampleChain<State, Kernel:TransitionKernel>(_ num_results:Int,
                                                        _ init_state:State,
                                                        _ kernel:Kernel,
                                                        _ num_burnin_steps:Int) -> Array<State>
where Kernel.State==State {
  // Initialize kernel side information
  var results = kernel.bootstrap_results(init_state)
  
  // Allocate result
  var states : [State] = [init_state]
  states.reserveCapacity(num_burnin_steps + num_results)
  
  // Run sampler
  for _ in 1..<num_burnin_steps + num_results {
    let (next_state, new_results) = kernel.one_step(states.last!, results)
    states.append(next_state)
    results = new_results
  }
  
  // Return only last num_results
  return Array(states.dropFirst(num_burnin_steps))
}

