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

import _Differentiation
import Benchmark
import SwiftFusion

let pose3SLAM = BenchmarkSuite(name: "Pose3SLAM") { suite in 
  let sphere2500URL = try! cachedDataset("sphere2500.g2o")
  let sphere2500Dataset =  try! G2OReader.G2OFactorGraph(g2oFile3D: sphere2500URL)

  // Uses `FactorGraph` on the sphere2500 dataset.
  // The solvers are configured to run for a constant number of *LM steps*, except when the LM solver is
  // unable to progress even with maximum lambda.
  // The linear solver is 200 iterations of CGLS.
  suite.benchmark(
    "FactorGraph, sphere2500, 30 LM steps, 200 CGLS steps",
    settings: Iterations(1), TimeUnit(.ms)
  ) {
    var val = sphere2500Dataset.initialGuess
    var graph = sphere2500Dataset.graph
    
    graph.store(PriorFactor(TypedID(0), Pose3()))
    
    var optimizer = LM()
    optimizer.max_iteration = 30
    optimizer.max_inner_iteration = 200
    
    do {
      try optimizer.optimize(graph: graph, initial: &val)
    } catch let error {
      print("The solver gave up, message: \(error.localizedDescription)")
    }
  }

  suite.benchmark(
    "sphere2500, chordal initialization",
    settings: Iterations(1), TimeUnit(.ms)
  ) {
    _ = ChordalInitialization.GetInitializations(
      graph: sphere2500Dataset.graph, ids: sphere2500Dataset.variableId)
  }

  let sphere2500DatasetChordal = try! G2OReader.G2OFactorGraph(
    g2oFile3D: sphere2500URL, chordal: true)

  suite.benchmark(
    "sphere2500, chordal graph, 1 LM step, 200 CGLS steps",
    settings: Iterations(1), TimeUnit(.ms)
  ) {
    var val = sphere2500DatasetChordal.initialGuess
    var graph = sphere2500DatasetChordal.graph
    
    graph.store(PriorFactor(TypedID(0), Pose3()))
    
    var optimizer = LM()
    optimizer.max_iteration = 1
    optimizer.max_inner_iteration = 200
    
    do {
      try optimizer.optimize(graph: graph, initial: &val)
    } catch let error {
      print("The solver gave up, message: \(error.localizedDescription)")
    }
  }
}
