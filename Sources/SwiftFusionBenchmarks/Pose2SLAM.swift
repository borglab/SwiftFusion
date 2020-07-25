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
    try! G2OReader.G2OFactorGraph(g2oFile2D: try! cachedDataset("input_INTEL_g2o.txt"))
  check(
    intelDataset.graph.error(at: intelDataset.initialGuess),
    near: 0.5 * 73565.64,
    accuracy: 1e-2)

  // Uses `FactorGraph` on the Intel dataset.
  // The solvers are configured to run for a constant number of steps.
  // The nonlinear solver is 10 iterations of Gauss-Newton.
  // The linear solver is 500 iterations of CGLS.
  suite.benchmark(
    "FactorGraph, Intel, 10 Gauss-Newton steps, 500 CGLS steps",
    settings: Iterations(1), TimeUnit(.ms)
  ) {
    var x = intelDataset.initialGuess
    var graph = intelDataset.graph
    graph.store(PriorFactor(TypedID(0), Pose2(0, 0, 0)))

    for _ in 0..<10 {
      let linearized = graph.linearized(at: x)
      var dx = x.tangentVectorZeros
      var optimizer = GenericCGLS(precision: 0, max_iteration: 500)
      optimizer.optimize(gfg: linearized, initial: &dx)
      x.move(along: dx)
    }

    check(graph.error(at: x), near: 0.5 * 0.987, accuracy: 1e-2)
  }
}

func check(_ actual: Double, near expected: Double, accuracy: Double) {
  if abs(actual - expected) > accuracy {
    print("ERROR: \(actual) != \(expected) (accuracy \(accuracy))")
    fatalError()
  }
}
