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

import _Differentiation
import Benchmark
import SwiftFusion
// import TensorFlow

let ppcaTrackingBenchmark = BenchmarkSuite(name: "PPCATracking") { suite in
  /// Returns a factor graph with a single `PPCATrackingFactor` with randomly initialized
  /// parameters.
  func makeFactorGraphWithOnePPCATrackingFactor() -> (FactorGraph, VariableAssignments) {
    var x = VariableAssignments()
    let poseId = x.store(Pose2(100, 100, 0))
    let latentId = x.store(Vector5.zero)

    var fg = FactorGraph()
    fg.store(PPCATrackingFactor.testFixture(poseId, latentId, seed: (1, 1)))
    return (fg, x)
  }

  /// Measures how long it takes to linearize a `PPCATrackingFactor`.
  suite.benchmark(
    "linearize PPCATrackingFactor",
    settings: Iterations(1), TimeUnit(.ms)
  ) { state in
    let (fg, x) = makeFactorGraphWithOnePPCATrackingFactor()
    try state.measure {
      _ = fg.linearized(at: x)
    }
  }

  /// Measures how long it takes to run CGLS on the lineraization of a `PPCATrackingFactor`.
  suite.benchmark(
    "cgls LinearizedPPCATrackingFactor",
    settings: Iterations(1), TimeUnit(.ms)
  ) { state in
    let (fg, x) = makeFactorGraphWithOnePPCATrackingFactor()
    let gfg = fg.linearized(at: x)
    try state.measure {
      var optimizer = GenericCGLS()
      var dx = x.tangentVectorZeros
      optimizer.optimize(gfg: gfg, initial: &dx)
    }
  }
}
