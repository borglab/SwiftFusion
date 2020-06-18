import Foundation
import TensorFlow
import XCTest

import PenguinStructures
@testable import SwiftFusion

/// A factor on two discrete labels evaluation the transition probability
struct DiscreteTransitionFactor {
  /// The number of states.
  let stateCount: Int

  /// Entry `i * stateCount + j` is the probability of transitioning from state `j` to state `i`.
  let transitionMatrix: [Double]

  init(
    _ inputId1: TypedID<Int, Int>,
    _ inputId2: TypedID<Int, Int>,
    _ stateCount: Int,
    _ transitionMatrix: [Double]
  ) {
    precondition(transitionMatrix.count == stateCount * stateCount)
    self.stateCount = stateCount
    self.transitionMatrix = transitionMatrix
  }

  func error(at label1: Int, _ label2: Int) -> Double {
    return -log(transitionMatrix[label2 * stateCount + label1])
  }
}

/// A factor with a switchable motion model.
///
/// `JacobianRows` specifies the `Rows` parameter of the Jacobian of this factor. See the
/// documentation on `NewJacobianFactor.jacobian` for more information. Use the typealiases below to
/// avoid specifying this type parameter every time you create an instance.
public struct NewSwitchingBetweenFactor<Pose: LieGroup, JacobianRows: FixedSizeArray>:
  NewLinearizableFactor
  where JacobianRows.Element == Tuple2<Pose.TangentVector, Pose.TangentVector>
{
  public typealias Variables = Tuple3<Pose, Int, Pose>

  public let edges: Variables.Indices
  
  /// Movement temmplates for each label.
  let movements: [Pose]

  public init(_ from: TypedID<Pose, Int>,
              _ label: TypedID<Int, Int>,
              _ to: TypedID<Pose, Int>,
              _ movements: [Pose]) {
    self.edges = Tuple3(from, label, to)
    self.movements = movements
  }

  public typealias ErrorVector = Pose.TangentVector

  @differentiable(wrt: (start, end))
  public func errorVector(_ start: Pose, _ label: Int, _ end: Pose) -> ErrorVector {
    let actualMotion = between(start, end)
    return movements[label].localCoordinate(actualMotion)
  }

  // Note: All the remaining code in this factor is boilerplate that we can eventually eliminate
  // with sugar.
  
  public func error(at x: Variables) -> Double {
    return errorVector(at: x).squaredNorm
  }

  public func errorVector(at x: Variables) -> Pose.TangentVector {
    return errorVector(x.head, x.tail.head, x.tail.tail.head)
  }

  public typealias Linearization = NewJacobianFactor<JacobianRows, ErrorVector>
  public func linearized(at x: Variables) -> Linearization {
    Linearization(linearizing: errorVector, at: x, edges: edges)
  }
}

/// A between factor on `Pose2`.
public typealias NewSwitchingBetweenFactor2 = NewSwitchingBetweenFactor<Pose2, Array3<Tuple2<Vector3, Vector3>>>

/// A between factor on `Pose3`.
public typealias NewSwitchingBetweenFactor3 = NewSwitchingBetweenFactor<Pose3, Array6<Tuple2<Vector6, Vector6>>>

class Scratch: XCTestCase {
  func createTrackingFactorGraph() -> (NewFactorGraph, VariableAssignments) {
    var variables = VariableAssignments()
    let x1 = variables.store(Pose2(0, 0, 0))
    let x2 = variables.store(Pose2(0, 0, 0))
    let x3 = variables.store(Pose2(0, 0, 0))

    var graph = NewFactorGraph()
    graph.store(NewPriorFactor2(x1, Pose2(10, 0, 0)))
    graph.store(NewPriorFactor2(x1, Pose2(10, 0, 0)))
    graph.store(NewPriorFactor2(x2, Pose2(11, 0, 0)))
    graph.store(NewPriorFactor2(x3, Pose2(12, 0, 0)))
    graph.store(NewBetweenFactor2(x1, x2, Pose2(1, 0, 0)))
    graph.store(NewBetweenFactor2(x2, x3, Pose2(1, 0, 0)))

    return (graph, variables)
  }

  /// Tracking example from Figure 2.a
  func testTrackingExample() {
    // create a factor graph
    var (graph, variables) = createTrackingFactorGraph()
    _ = graph as NewFactorGraph

    XCTAssertEqual(graph.storage.count, 2)

    // optimize
    var opt = LM()
    try! opt.optimize(graph: graph, initial: &variables)

    // check results
    print(variables[TypedID<Pose2, Int>(0)])
    print(variables[TypedID<Pose2, Int>(1)])
    print(variables[TypedID<Pose2, Int>(2)])
  }

  func createSwitchingFactorGraph() -> (NewFactorGraph, VariableAssignments) {
    var variables = VariableAssignments()
    let x1 = variables.store(Pose2(0, 0, 0))
    let x2 = variables.store(Pose2(0, 0, 0))
    let x3 = variables.store(Pose2(0, 0, 0))
    let q1 = variables.store(0)
    let q2 = variables.store(0)

    // Model parameters.
    let labelCount = 3
    let transitionMatrix: [Double] = [
      0.8, 0.1, 0.1,
      0.1, 0.8, 0.1,
      0.1, 0.1, 0.8
    ]
    let movements = [
      Pose2(1, 0, 0),       // go forwards
      Pose2(1, 0, .pi / 4), // turn left
      Pose2(1, 0, -.pi / 4)  // turn right
    ]

    var graph = NewFactorGraph()
    graph.store(NewPriorFactor2(x1, Pose2(10, 0, 0)))
    graph.store(NewPriorFactor2(x1, Pose2(10, 0, 0)))
    graph.store(NewPriorFactor2(x2, Pose2(11, 0, 0)))
    graph.store(NewPriorFactor2(x3, Pose2(12, 0, 0)))
    graph.store(NewSwitchingBetweenFactor2(x1, q1, x2, movements))
    graph.store(NewSwitchingBetweenFactor2(x2, q2, x3, movements))
    graph.store(DiscreteTransitionFactor(q1, q2, labelCount, transitionMatrix))

    return (graph, variables)
  }

  /// Tracking switching from Figure 2.b
  func testSwitchingExample() {
    // create a factor graph
    var (graph, variables) = createSwitchingFactorGraph()
    _ = graph as NewFactorGraph

    XCTAssertEqual(graph.storage.count, 2)

    // optimize
    var opt = LM()
    try! opt.optimize(graph: graph, initial: &variables)

    // check results
    print(variables[TypedID<Pose2, Int>(0)])
    print(variables[TypedID<Pose2, Int>(1)])
    print(variables[TypedID<Pose2, Int>(2)])
  }
}
