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

import _Differentiation
import Benchmark
import SwiftFusion
import TensorFlow

let patchBenchmark = BenchmarkSuite(name: "Patch") { suite in
  suite.benchmark(
    "PatchForward",
    settings: Iterations(1), TimeUnit(.ms)
  ) {
    let rows = 100
    let cols = 100
    let image = Tensor<Double>(randomNormal: [500, 500, 1])
    _ = image.patch(at: OrientedBoundingBox(center: Pose2(100, 100, 0.5), rows: rows, cols: cols))
  }

  suite.benchmark(
    "PatchJacobian",
    settings: Iterations(1), TimeUnit(.ms)
  ) {
    let rows = 28
    let cols = 62
    let latentDimension = 5

    let image = Tensor<Double>(randomNormal: [500, 500])

    let W = Tensor<Double>(randomNormal: [rows * cols, latentDimension])
    let mu = Tensor<Double>(randomNormal: [rows, cols])
    func errorVector(_ center: Pose2, _ latent: Vector5) -> Tensor<Double> {
      let bbox = OrientedBoundingBox(center: center, rows: rows, cols: cols)
      let generated = mu + matmul(W, latent.flatTensor.expandingShape(at: 1)).reshaped(to: [rows, cols])
      return generated - image.patch(at: bbox)
    }

    let (value, pb) = valueWithPullback(at: Pose2(100, 100, 0.5), Vector5.zero, in: errorVector)

    for i in 0..<rows {
      for j in 0..<cols {
        var basisVector = Tensor<Double>(zeros: [rows, cols])
        basisVector[i, j] = Tensor(1)
        _ = pb(basisVector)
      }
    }
  }
}
