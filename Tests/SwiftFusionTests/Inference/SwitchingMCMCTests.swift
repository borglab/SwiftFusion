//
//  SwitchingMCMCTests.swift
//
//
//  Frank Dellaert and Marc Rasi
//  July 2020

import Foundation
import TensorFlow
import XCTest

import PenguinStructures
@testable import SwiftFusion

/// A between factor on `Pose2`.
public typealias SwitchingBetweenFactor2 = SwitchingBetweenFactor<Pose2>

/// A between factor on `Pose3`.
public typealias SwitchingBetweenFactor3 = SwitchingBetweenFactor<Pose3>

class Scratch: XCTestCase {
  let origin = Pose2(0,0,0)
  let forwardMove = Pose2(1,0,0)
  let (z1, z2, z3) = (Pose2(10, 0, 0),Pose2(11, 0, 0),Pose2(12, 0, 0))
  var expectedTrackingError : Double = 0.0
  
  // Calculate error by hand in test fixture
  override func setUp() {
    super.setUp()
    expectedTrackingError = 0.5*(2 * origin.localCoordinate(z1).squaredNorm
      +  origin.localCoordinate(z2).squaredNorm
      +  origin.localCoordinate(z3).squaredNorm
      +  2 * origin.localCoordinate(forwardMove).squaredNorm)
  }
  
  /// Create an example tracking graph
  func createTrackingFactorGraph() -> (FactorGraph, VariableAssignments) {
    // First get 3 pose variables
    var variables = VariableAssignments()
    let x1 = variables.store(origin)
    let x2 = variables.store(origin)
    let x3 = variables.store(origin)
    
    // Use ids x1,x2,x3 to create factors
    var graph = FactorGraph()
    graph.store(PriorFactor(x1, z1)) // prior
    graph.store(PriorFactor(x1, z1))
    graph.store(PriorFactor(x2, z2))
    graph.store(PriorFactor(x3, z3))
    graph.store(BetweenFactor(x1, x2, forwardMove))
    graph.store(BetweenFactor(x2, x3, forwardMove))
    
    return (graph, variables)
  }
  
  /// Just a helper for debugging
  func printPoses(_ variables : VariableAssignments) {
    print(variables[TypedID<Pose2>(0)])
    print(variables[TypedID<Pose2>(1)])
    print(variables[TypedID<Pose2>(2)])
  }
  
  /// Tracking example from Figure 2.a in Annual Reviews paper
  func testTrackingExample() {
    // create a factor graph
    var (graph, variables) = createTrackingFactorGraph()
    _ = graph as FactorGraph //
    
    // check number of factor types
    XCTAssertEqual(graph.storage.count, 2)
    
    // check error at initial estimate
    XCTAssertEqual(graph.error(at: variables), expectedTrackingError)
    
    // optimize
    var opt = LM()
    try! opt.optimize(graph: graph, initial: &variables)
    
    // print
    printPoses(variables)
  }
  
  /// Switching system example from Figure 2.b in that same paper
  func createSwitchingFactorGraph() -> (FactorGraph, VariableAssignments) {
    var variables = VariableAssignments()
    let x1 = variables.store(origin)
    let x2 = variables.store(origin)
    let x3 = variables.store(origin)
    
    // We now have discrete labels, as well
    let q1 = variables.store(0)
    let q2 = variables.store(0)
    
    // Model parameters include a 3x3 transition matrix and 3 motion models.
    let labelCount = 3
    let transitionMatrix: [Double] = [
      0.8, 0.1, 0.1,
      0.1, 0.8, 0.1,
      0.1, 0.1, 0.8
    ]
    let movements = [
      forwardMove,          // go forwards
      Pose2(1, 0, .pi / 4), // turn left
      Pose2(1, 0, -.pi / 4) // turn right
    ]
    
    // Create the graph itself
    var graph = FactorGraph()
    graph.store(PriorFactor(x1, z1)) // prior
    graph.store(PriorFactor(x1, z1))
    graph.store(PriorFactor(x2, z2))
    graph.store(PriorFactor(x3, z3))
    graph.store(SwitchingBetweenFactor2(x1, q1, x2, movements))
    graph.store(SwitchingBetweenFactor2(x2, q2, x3, movements))
    graph.store(DiscreteTransitionFactor(q1, q2, labelCount, transitionMatrix))
    
    return (graph, variables)
  }
  
  /// Just a helper for debugging
  func printLabels(_ variables : VariableAssignments) {
    print(variables[TypedID<Int>(0)])
    print(variables[TypedID<Int>(1)])
  }

  /// Returns the gradient of the error function of `graph` at `x`.
  func errorGradient(_ graph: FactorGraph, _ x: VariableAssignments) -> AllVectors {
    let l = graph.linearized(at: x)
    return l.errorVectors_linearComponent_adjoint(l.errorVectors(at: x.tangentVectorZeros))
  }

  func printGradient(_ grad: AllVectors) {
    for i in 0..<3 {
      print("  \(grad[TypedID<Vector3>(i)])")
    }
  }
  
  /// Tracking switching from Figure 2.b
  func testSwitchingExample() {
    // create a factor graph
    var (graph, variables) = createSwitchingFactorGraph()
    _ = graph as FactorGraph
    _ = variables as VariableAssignments
    
    // check number of factor types
    XCTAssertEqual(graph.storage.count, 3)
    
    // check error at initial estimate, allow slack to account for discrete transition
    XCTAssertEqual(graph.error(at: variables), 234, accuracy:0.3)
    
    // optimize
    var opt = LM()
    try! opt.optimize(graph: graph, initial: &variables)
    
    
    // print
    // printLabels(variables)
    // printPoses(variables)

    // Create initial state for MCMC sampler
    let current_state = variables
    
    // Do MCMC the tfp way
    let num_results = 50
    let num_burnin_steps = 30
    
    /// Proposal to change one label, and re-optimize
    let flipAndOptimize = {(x:VariableAssignments, r: inout AnyRandomNumberGenerator) -> VariableAssignments in
      let labelVars = x.storage[ObjectIdentifier(Int.self)]
      //  let positionVars = x.storage[ObjectIdentifier(Pose2.self)]
      
      // Randomly change one label.
      let i = Int.random(in: 0..<labelVars!.count, using: &r)
      let id = TypedID<Int>(i)
      var y = x
      y[id] = Int.random(in: 0..<3)
      
      // Pose2SLAM to find new proposed positions.
      // print("Pose2SLAM starting at:")
      // self.printLabels(y)
      // self.printPoses(y)
      do {
        try opt.optimize(graph: graph, initial: &y)
      } catch {
        // TODO: Investigate why the optimizer fails to make progress when it's near the solution.
        // print("ignoring optimizer error")
      }

      // Check that we found a local minimum by asserting that the gradient is 0. The optimizer
      // isn't good at getting precisely to the minimum, so the assertion has a pretty big
      // `accuracy` argument.
      let grad = self.errorGradient(graph, y)
      XCTAssertEqual(grad.squaredNorm, 0, accuracy: 0.1)

      // print("Pose2SLAM solution:")
      // self.printLabels(y)
      // self.printPoses(y)
      // print("Gradient of error function at solution:")
      // self.printGradient(grad)
      return y
    }
    
    let kernel = RandomWalkMetropolis(
      target_log_prob_fn: {(x:VariableAssignments) in 0.0},
      new_state_fn: flipAndOptimize
    )
    
    let states = sampleChain(
      num_results,
      current_state,
      kernel,
      num_burnin_steps
    )
    _ = states as Array
    XCTAssertEqual(states.count, num_results)
  }
}
