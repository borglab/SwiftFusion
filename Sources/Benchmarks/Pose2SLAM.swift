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

  let intelDataset =
    try! G2OReader.G2ONonlinearFactorGraph(g2oFile2D: try! cachedDataset("input_INTEL_g2o.txt"))
  check(intelDataset.graph.error(intelDataset.initialGuess), near: 73565.64, accuracy: 1e-2)

  // Uses `NonlinearFactorGraph` on the Intel dataset.
  // The solvers are configured to run for a constant number of steps.
  // The nonlinear solver is 5 iterations of Gauss-Newton.
  // The linear solver is 100 iterations of CGLS.
  suite.benchmark(
    "NonlinearFactorGraph, Intel, 5 Gauss-Newton steps, 100 CGLS steps",
    settings: .iterations(1)
  ) {
    var val = intelDataset.initialGuess
    for _ in 0..<5 {
      let gfg = intelDataset.graph.linearize(val)
      let optimizer = CGLS(precision: 0, max_iteration: 100)
      var dx = VectorValues()
      for i in 0..<val.count {
        dx.insert(i, Vector(zeros: 3))
      }
      optimizer.optimize(gfg: gfg, initial: &dx)
      val.move(along: dx)
    }
//    check(intelDataset.graph.error(val), near: 35.59, accuracy: 1e-2)
  }

  let intelDatasetGen =
    try! G2OReader.G2OGenericFactorGraph(g2oFile2D: try! cachedDataset("input_INTEL_g2o.txt"))
  check(
    intelDatasetGen.graph.error(at: intelDatasetGen.initialGuess),
    near: 73565.64,
    accuracy: 1e-2)

  // Uses `GenericFactorGraph` on the Intel dataset.
  // The solvers are configured to run for a constant number of steps.
  // The nonlinear solver is 10 iterations of Gauss-Newton.
  // The linear solver is 500 iterations of CGLS.
  suite.benchmark(
    "GenericFactorGraph, Intel, 10 Gauss-Newton steps, 500 CGLS steps",
    settings: .iterations(1)
  ) {
    var x = intelDatasetGen.initialGuess
    var graph = intelDatasetGen.graph
    graph.store(GenericPriorFactor2(TypedID(0), Pose2(0, 0, 0)))

    for _ in 0..<10 {
      let linearized = graph.linearized(at: x)
      var dx = x.linearizedZero
      var optimizer = GenericCGLS(precision: 0, max_iteration: 500)
      optimizer.optimize(gfg: linearized, initial: &dx)
      x.move(along: (-1) * dx)
    }

    check(graph.error(at: x), near: 0.987, accuracy: 1e-2)
  }
}

func check(_ actual: Double, near expected: Double, accuracy: Double) {
  if abs(actual - expected) > accuracy {
    print("ERROR: \(actual) != \(expected) (accuracy \(accuracy))")
    fatalError()
  }
}
