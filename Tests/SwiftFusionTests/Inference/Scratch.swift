import Foundation
import TensorFlow
import XCTest

import PenguinStructures
@testable import SwiftFusion

class Scratch: XCTestCase {
  func createTrackingFactorGraph() -> (NewFactorGraph, VariableAssignments) {
    var variables = VariableAssignments()
    let keys = repeatElement(Pose2(0, 0, 0), count: 3).map { variables.store($0) }

    let measurements = [
      Pose2(10, 0, 0),
      Pose2(11, 0, 0),
      Pose2(12, 0, 0)
    ]

    var graph = NewFactorGraph()
    graph.store(NewPriorFactor2(keys[0], Pose2(10, 0, 0)))
    for i in 0..<3 {
      graph.store(NewPriorFactor2(keys[i], measurements[i]))
    }
    graph.store(NewBetweenFactor2(keys[0], keys[1], Pose2(1, 0, 0)))
    graph.store(NewBetweenFactor2(keys[1], keys[2], Pose2(1, 0, 0)))

    return (graph, variables)
  }

  func testWholeEnchilada() {
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
}
