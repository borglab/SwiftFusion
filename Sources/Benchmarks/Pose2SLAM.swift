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

/// Benchmarks Pose2SLAM solutions.

import Benchmark
import SwiftFusion

let pose2SLAM = BenchmarkSuite(name: "Pose2SLAM") { suite in

  let intelNonlinearFactorGraph = try! G2OFactorGraph(fromG2O: try! cachedDataset("input_INTEL_g2o.txt"))
  check(intelNonlinearFactorGraph.graph.error(intelNonlinearFactorGraph.initialGuess), near: 73565.64, accuracy: 1e-2)

  // Uses `NonlinearFactorGraph` on the Intel dataset.
  // The solvers are configured to run for a constant number of steps.
  // The nonlinear solver is 5 iterations of Gauss-Newton.
  // The linear solver is 100 iterations of CGLS.
  suite.benchmark(
    "NonlinearFactorGraph, Intel, 10 Gauss-Newton steps, 500 CGLS steps",
    settings: .iterations(1)
  ) {
    var graph = intelNonlinearFactorGraph.graph
    graph += PriorFactor(0, Pose2(0, 0, 0))
    var val = intelNonlinearFactorGraph.initialGuess
    for _ in 0..<10 {
      let gfg = graph.linearize(val)
      let optimizer = CGLS(precision: 0, max_iteration: 500)
      var dx = VectorValues()
      for i in 0..<val.count {
        dx.insert(i, Vector(zeros: 3))
      }
      optimizer.optimize(gfg, initial: &dx)
      val.move(along: dx)
      print(graph.error(val))
    }
    //check(graph.error(val), near: 63.55, accuracy: 1e-2)
  }

  let intelPoseSLAMFactorGraph = try! G2OPoseSLAMFactorGraph(fromG2O: try! cachedDataset("input_INTEL_g2o.txt"))
  check(intelPoseSLAMFactorGraph.graph.error(at: intelPoseSLAMFactorGraph.initialGuess), near: 73565.64, accuracy: 1e-2)

  // Uses `PoseSLAMFactorGraph` on the Intel dataset.
  // The solvers are configured to run for a constant number of steps.
  // The nonlinear solver is 5 iterations of Gauss-Newton.
  // The linear solver is 100 iterations of CGLS.
  suite.benchmark(
    "PoseSLAMFactorGraph, Intel, 5 Gauss-Newton steps, 100 CGLS steps",
    settings: .iterations(1)
  ) {
    var graph = intelPoseSLAMFactorGraph.graph
    graph.priors.parameters.append(Pose2(0, 0, 0))
    graph.priorInputs.append(0)
    var val = intelPoseSLAMFactorGraph.initialGuess
    for _ in 0..<10 {
      let gfg = graph.linearized(at: val)
      let optimizer = CGLS(precision: 0, max_iteration: 500)
      var dx = gfg.zeroInput
      optimizer.optimize(gfg, initial: &dx)
      for index in val.indices {
        val[index].move(along: dx[index])
      }
      print(graph.error(at: val))
    }
    //check(graph.error(at: val), near: 63.55, accuracy: 1e-2)
  }

  suite.benchmark("GenericFactorGraphExample", settings: .iterations(1)) {
    runSimplePose2SLAM()
  }

  suite.benchmark("GenericFactorGraphPose2SLAM", settings: .iterations(1)) {
    runGenericFactorGraphBenchmark()
  }

}

/// Builds an initial guess and a factor graph from a g2o file.
struct G2OFactorGraph: G2OReader {
  /// The initial guess.
  var initialGuess: Values = Values()

  /// The factor graph representing the measurements.
  var graph: NonlinearFactorGraph = NonlinearFactorGraph()

  public mutating func addInitialGuess(index: Int, pose: Pose2) {
    initialGuess.insert(index, pose)
  }

  public mutating func addMeasurement(frameIndex: Int, measuredIndex: Int, pose: Pose2) {
    graph += BetweenFactor(frameIndex, measuredIndex, pose)
  }
}

/// Builds an initial guess and a factor graph from a g2o file.
struct G2OPoseSLAMFactorGraph: G2OReader {
  /// The initial guess.
  var initialGuess: [Pose2] = []

  /// The factor graph representing the measurements.
  var graph: Pose2SLAMFactorGraph = Pose2SLAMFactorGraph()

  public mutating func addInitialGuess(index: Int, pose: Pose2) {
    precondition(index == initialGuess.count)
    initialGuess.append(pose)
  }

  public mutating func addMeasurement(frameIndex: Int, measuredIndex: Int, pose: Pose2) {
    graph.betweens.parameters.append(pose)
    graph.betweenInputs.append((frameIndex, measuredIndex))
  }
}

func check(_ actual: Double, near expected: Double, accuracy: Double) {
  if abs(actual - expected) > accuracy {
    print("ERROR: \(actual) != \(expected) (accuracy \(accuracy))")
    fatalError()
  }
}
