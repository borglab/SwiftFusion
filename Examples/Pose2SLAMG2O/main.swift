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

/// Loads a g2o file into a factor graph and then runs inference on the factor graph.
///
/// See https://lucacarlone.mit.edu/datasets/ for g2o specification and example datasets.
///
/// Usage: Pose2SLAMG2O [path to .g2o file]
///
/// Missing features:
/// - Does not take g2o information matrix into account.
/// - Does not use a proper general purpose solver.
/// - Has not been compared against other implementations, so it could be wrong.

import Foundation
import SwiftFusion
import TensorFlow

func main() {
  // Parse commandline.
  guard CommandLine.arguments.count == 2 else {
    print("Usage: Pose2SLAMG2O [path to .g2o file]")
    return
  }
  let g2oURL = URL(fileURLWithPath: CommandLine.arguments[1])

  // Load .g2o file.
  var problem = try! G2OReader.G2ONonlinearFactorGraph(g2oFile2D: g2oURL)

  // Add prior on the pose with key 0.
  problem.graph += OldPriorFactor(0, Pose2(0, 0, 0))

  // Run inference.
  // TODO: Change this to use a general purpose solver instead of iterating ourselves, when a
  // general purpose solver exists.
  var val = problem.initialGuess
  print("Initial error: \(problem.graph.error(val))")
  for _ in 0..<10 {
    let gfg = problem.graph.linearize(val)
    let optimizer = CGLS(precision: 1e-6, max_iteration: 200)
    var dx = VectorValues()
    for i in 0..<val.count {
      dx.insert(i, Vector(zeros: 3))
    }
    optimizer.optimize(gfg: gfg, initial: &dx)
    val.move(along: dx)
    print("Current error: \(problem.graph.error(val))")
  }
  print("Final error: \(problem.graph.error(val))")
}

main()
