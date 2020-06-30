//
//  MCMCTests.swift
//  SwiftFusionTests
//
//  Created by Frank Dellaert on 6/23/20.
//

import TensorFlow
import SwiftFusion

import XCTest

class MCMCTests: XCTestCase {
  
  /// Sampling from the Standard Normal Distribution.
  /// Inspired by testRWM1DUniform from tfp
  func testRWM1DUniform() throws {
    // Create kernel for MCMC sampler:
    // target_log_prob_fn is proportional to log p(x), where p is a zero-mean Gaussian
    // new_state_fn is a symmetric (required for Metropolis) proposal density,
    // in this case perturbing x with uniformly sampled perturbations.
    let kernel = RandomWalkMetropolis(
      target_log_prob_fn: {(x:Double) in -0.5*x*x}, //  tfd.Normal(loc=dtype(0), scale=dtype(1))
      new_state_fn: {(x:Double) in x + Double.random(in: -1..<1)}
    )
    
    // Run the sampler for 2500 steps, discarding the first 500 asa burn-in
    let num_results = 2000
    let initial_state: Double = 1.0
    let num_burnin_steps = 500
    let samples = sampleChain(num_results, initial_state, kernel, num_burnin_steps)
    
    // Assert samples have the right type and size
    _ = samples as [Double]
    XCTAssertEqual(samples.count, num_results)
    
    // Check the mean and standard deviation, which should be 0 and 1, respectively
    let tensor = Tensor<Double>(samples)
    let sample_mean = tensor.mean().scalarized()
    let sample_std = tensor.standardDeviation().scalarized()
    XCTAssertEqual(0, sample_mean, accuracy: 0.17)
    XCTAssertEqual(1, sample_std, accuracy: 0.2)
  }

}
