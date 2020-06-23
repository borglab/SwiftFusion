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
