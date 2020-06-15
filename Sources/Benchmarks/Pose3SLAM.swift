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
  
  var gridDataset =  // try! G2OReader.G2ONewFactorGraph(g2oFile3D: try! cachedDataset("pose3example.txt"))
        try! G2OReader.G2ONewFactorGraph(g2oFile3D: try! cachedDataset("sphere2500.g2o"))
  //  check(gridDataset.graph.error(gridDataset.initialGuess), near: 12.99, accuracy: 1e-2)
  
  // Uses `NonlinearFactorGraph` on the Intel dataset.
  // The solvers are configured to run for a constant number of steps.
  // The nonlinear solver is 5 iterations of Gauss-Newton.
  // The linear solver is 100 iterations of CGLS.
  suite.benchmark(
    "NewFactorGraph, sphere2500, 30 LM steps, max 6 G-N steps, 200 CGLS steps",
    settings: Iterations(1)
  ) {
    var val = gridDataset.initialGuess
    var graph = gridDataset.graph
    
    graph.store(NewPriorFactor3(TypedID(0), Pose3(Rot3.fromTangent(Vector3.zero), Vector3.zero)))
    
    var optimizer = LM()
    
    do {
      try optimizer.optimize(graph: graph, initial: &val)
    } catch let error {
      print("The solver gave up, message: \(error.localizedDescription)")
    }
  }
}
