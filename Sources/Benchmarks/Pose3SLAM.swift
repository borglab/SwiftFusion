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

/// Benchmarks Pose3SLAM solutions.

import Benchmark
import SwiftFusion

let pose3SLAM = BenchmarkSuite(name: "Pose3SLAM") { suite in

  var gridDataset =
    try! G2OReader.G2ONonlinearFactorGraph(g2oFile3D: try! cachedDataset("pose3example.txt"))
  check(gridDataset.graph.error(gridDataset.initialGuess), near: 12.99, accuracy: 1e-2)

  // Uses `NonlinearFactorGraph` on the Intel dataset.
  // The solvers are configured to run for a constant number of steps.
  // The nonlinear solver is 5 iterations of Gauss-Newton.
  // The linear solver is 100 iterations of CGLS.
  suite.benchmark(
    "NonlinearFactorGraph, Pose3Example, 50 Gauss-Newton steps, 200 CGLS steps",
    settings: .iterations(1)
  ) {
    var val = gridDataset.initialGuess
    gridDataset.graph += PriorFactor(0, Pose3())
    for _ in 0..<50 {
      let gfg = gridDataset.graph.linearize(val)
      let optimizer = CGLS(precision: 0, max_iteration: 100)
      var dx = VectorValues()
      for i in 0..<val.count {
        dx.insert(i, Vector(zeros: 6))
      }
      optimizer.optimize(gfg: gfg, initial: &dx)
      val.move(along: dx)
    }
    for i in val.keys.sorted() {
      print(val[i, as: Pose3.self].t)
    }
  }
}
