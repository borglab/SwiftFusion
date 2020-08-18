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

import Benchmark
import SwiftFusion
import TensorFlow

let patchBenchmark = BenchmarkSuite(name: "Patch") { suite in
  /// Measures speed of taking a 100x100x1 patch from a 500x500x1 image.
  suite.benchmark(
    "100x100x1 patch from 500x500x1 image",
    settings: Iterations(1), TimeUnit(.ms)
  ) {
    let rows = 100
    let cols = 100
    let image = Tensor<Double>(randomNormal: [500, 500, 1])
    _ = image.patch(at: OrientedBoundingBox(center: Pose2(100, 100, 0.5), rows: rows, cols: cols))
  }
}
