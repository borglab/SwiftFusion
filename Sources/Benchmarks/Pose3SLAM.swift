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
  
  var gridDataset_old =
    try! G2OReader.G2ONonlinearFactorGraph(g2oFile3D: try! cachedDataset("pose3example.txt"))
  // Uses `NonlinearFactorGraph` on the GTSAM pose3example dataset.
  // The solvers are configured to run for a constant number of steps.
  // The nonlinear solver is 40 iterations of Gauss-Newton.
  // The linear solver is 200 iterations of CGLS.
  suite.benchmark(
    "NonlinearFactorGraph, Pose3Example, 40 Gauss-Newton steps, 200 CGLS steps",
    settings: Iterations(1)
  ) {
    var val = gridDataset_old.initialGuess
    gridDataset_old.graph += PriorFactor(0, Pose3())
    for _ in 0..<40 {
      print("error = \(gridDataset_old.graph.error(val))")
      let gfg = gridDataset_old.graph.linearize(val)
      let optimizer = CGLS(precision: 0, max_iteration: 200)
      var dx = VectorValues()
      for i in 0..<val.count {
        dx.insert(i, Vector(zeros: 6))
      }
      optimizer.optimize(gfg: gfg, initial: &dx)
      print("gfg error = \(gfg.residual(dx).norm)")
      val.move(along: dx)
    }
    for i in val.keys.sorted() {
      print(val[i, as: Pose3.self].t)
    }
  }
  
  var sphere2500Dataset =  try! G2OReader.G2ONewFactorGraph(g2oFile3D: try! cachedDataset("sphere2500.g2o"))
  // Uses `NewFactorGraph` on the GTSAM sphere2500 dataset.
  // The solvers are configured to run for a constant number of *LM steps*, except when the LM solver is
  // unable to progress even with maximum lambda.
  // The linear solver is 200 iterations of CGLS.
  suite.benchmark(
    "NewFactorGraph, sphere2500, 30 LM steps, 200 CGLS steps",
    settings: Iterations(1)
  ) {
    var val = sphere2500Dataset.initialGuess
    var graph = sphere2500Dataset.graph
    
    graph.store(NewPriorFactor3(TypedID(0), Pose3(Rot3.fromTangent(Vector3.zero), Vector3.zero)))
    
    var optimizer = LM()
    optimizer.verbosity = .SUMMARY
    optimizer.max_iteration = 30
    optimizer.max_inner_iteration = 200
    
    do {
      try optimizer.optimize(graph: graph, initial: &val)
    } catch let error {
      print("The solver gave up, message: \(error.localizedDescription)")
    }
  }
}
