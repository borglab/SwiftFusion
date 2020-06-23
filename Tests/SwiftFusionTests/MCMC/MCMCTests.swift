//
//  MCMCTests.swift
//  SwiftFusionTests
//
//  Created by Frank Dellaert on 6/23/20.
//

import TensorFlow

/// Minimal requirements to efficiently implement a Markov chain Monte Carlo (MCMC) transition kernel. A transition kernel returns a new state given some old state. It also takes (and returns) "side information" which may be used for debugging or optimization purposes (i.e, to "recycle" previously computed results).
/// Inspired by tpf [TransitionKernel](https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/TransitionKernel)
protocol TransitionKernel {
  associatedtype State
  associatedtype Results
  
  /// Takes one step of the TransitionKernel.
  func one_step(_ current_state: State, _ previous_kernel_results: Results) -> (State, Results)
  
  /// Returns an object with the same type as returned by one_step(...)[1].
  func bootstrap_results(_ init_state: State) -> Results
}

/// Implements Markov chain Monte Carlo via repeated TransitionKernel steps
/// Inspired by tfp [sample_chain](https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/sample_chain)
func sampleChain<State, Kernel:TransitionKernel>(_ num_results:Int,
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

/// Runs one step of the RWM algorithm with symmetric proposal.
/// Inspired by tfp [RandomWalkMetropolis](https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/RandomWalkMetropolis)
/// TODO(frank): currently does Double only, should be generic
class RandomWalkMetropolis<State> : TransitionKernel {
  typealias Results = Double // target_log_prob_fn of previously accepted state
  
  let target_log_prob_fn: (State) -> Double
  let new_state_fn : (State)->State
  
  init(target_log_prob_fn: @escaping (State) -> Double, new_state_fn : @escaping (State)->State) {
    self.target_log_prob_fn = target_log_prob_fn
    self.new_state_fn = new_state_fn
  }
  
  /// Runs one iteration of Random Walk Metropolis.
  /// TODO(frank): should this be done with inout params in Value-semantics world?
  func one_step(_ current_state: State, _ previous_kernel_results: Results) -> (State, Results) {
    
    // calculate next state, and new log probability
    let new_state = new_state_fn(current_state)
    let new_log_prob = target_log_prob_fn(new_state)
    
    // Calculate log of acceptance ratio p(x')/p(x) = log p(x') - log p(x)
    let current_log_prob = previous_kernel_results
    let log_accept_ratio = new_log_prob - current_log_prob
    
    // If p(x')/log p(x) >= 1 , i.e., log_accept_ratio >= 0, we always accept
    // otherwise we accept randomly with probability p(x')/log p(x).
    // We do this by randomly sampling u from [0,1], and comparing log(u) with log_accept_ratio.
    let u = Double.random(in: 0..<1)
    if (log(u) <= log_accept_ratio) {
      return (new_state, new_log_prob)
    } else {
      return (current_state, current_log_prob)
    }
  }
  
  /// Initializes side information
  func bootstrap_results(_ init_state: State) -> Results {
    return target_log_prob_fn(init_state)
  }
}

import XCTest

class MCMCTests: XCTestCase {
  
  override func setUpWithError() throws {
    // Put setup code here. This method is called before the invocation of each test method in the class.
  }
  
  override func tearDownWithError() throws {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
  }
  
  /// Sampling from the Standard Normal Distribution.
  /// Inspired by testRWM1DUniform from tfp
  func testRWM1DUniform() throws {
    let kernel = RandomWalkMetropolis(
      target_log_prob_fn: {(x:Double) in -0.5*x*x}, //  tfd.Normal(loc=dtype(0), scale=dtype(1))
      new_state_fn: {(x:Double) in x + Double.random(in: -1..<1)}
    )
    
    let num_results = 2000
    let samples = sampleChain(num_results, 1.0, kernel, 500)
    _ = samples as [Double]
    XCTAssertEqual(samples.count, num_results)
    
    let tensor = Tensor<Double>(samples)
    let sample_mean = tensor.mean().scalarized()
    let sample_std = tensor.standardDeviation().scalarized()
    XCTAssertEqual(0, sample_mean, accuracy: 0.17)
    XCTAssertEqual(1, sample_std, accuracy: 0.2)
  }

}
