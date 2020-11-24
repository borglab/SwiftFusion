import _Differentiation
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

/// Minimal requirements to efficiently implement a Markov chain Monte Carlo (MCMC) transition kernel. A transition kernel returns a new state given some old state. It also takes (and returns) "side information" which may be used for debugging or optimization purposes (i.e, to "recycle" previously computed results).
/// Inspired by tpf [TransitionKernel](https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/TransitionKernel)
public protocol TransitionKernel {
  associatedtype State
  associatedtype Results
  
  /// Takes one step of the TransitionKernel.
  func one_step(_ current_state: State, _ previous_kernel_results: Results) -> (State, Results)
  
  /// Returns an object with the same type as returned by one_step(...)[1].
  func bootstrap_results(_ init_state: State) -> Results
}

