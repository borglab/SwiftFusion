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

import Foundation

struct FileHandlerOutputStream: TextOutputStream {
    private let fileHandle: FileHandle
    let encoding: String.Encoding

    init(_ fileHandle: FileHandle, encoding: String.Encoding = .utf8) {
        self.fileHandle = fileHandle
        self.encoding = encoding
    }

    mutating func write(_ string: String) {
        if let data = string.data(using: encoding) {
            fileHandle.write(data)
        }
    }
}

let pose3SLAM = BenchmarkSuite(name: "Pose3SLAM") { suite in

  var gridDataset = // try! G2OReader.G2ONonlinearFactorGraph(g2oFile3D: try! cachedDataset("pose3example.txt"))
    try! G2OReader.G2ONonlinearFactorGraph(g2oFile3D: try! cachedDataset("parking-garage.g2o"))
  
//  check(gridDataset.graph.error(gridDataset.initialGuess), near: 12.99, accuracy: 1e-2)

  // Uses `NonlinearFactorGraph` on the Intel dataset.
  // The solvers are configured to run for a constant number of steps.
  // The nonlinear solver is 5 iterations of Gauss-Newton.
  // The linear solver is 100 iterations of CGLS.
  suite.benchmark(
    "NonlinearFactorGraph, Pose3Example, 20 Gauss-Newton steps, 200 CGLS steps",
    settings: .iterations(1)
  ) {
    print("")
    var val = gridDataset.initialGuess
    gridDataset.graph += PriorFactor(0, Pose3())
    
    var old_error = gridDataset.graph.error(val)
    print("[LM OUTER] initial error = \(old_error)")
    
    var lambda = 1e-6
    for _ in 0..<20 { // outer loop
      print("[LM OUTER] outer loop start")
      let gfg = gridDataset.graph.linearize(val)
      let optimizer = CGLS(precision: 0, max_iteration: 200)
      var dx = VectorValues()
      for i in 0..<val.count {
        dx.insert(i, Vector(zeros: 6))
      }
      
      var inner_iter_step = 0
      var inner_success = false
      for _ in 0..<20 {
        print("[LM INNER] starting one iteration, lambda = \(lambda)")
        var damped = gfg
        
//        for i in dx.keys {
//          var j0 = [lambda * Matrix(eye: dx[i].dimension)]
//          var b0 = Vector(zeros: dx[i].dimension)
//          damped += JacobianFactor([i], j0, b0)
//        }
        
        var dx_t = dx
        optimizer.optimize(gfg: damped, initial: &dx_t)
        print("[LM INNER] damped error = \(damped.residual(dx_t).norm), lambda = \(lambda)")
        var oldval = val
        val.move(along: dx_t)
        let this_error = gridDataset.graph.error(val)
        let delta_error = old_error - this_error
        print("[LM INNER] error = \(this_error)")
        print("[LM INNER] delta = \(delta_error)")
        if delta_error > .ulpOfOne {
          old_error = this_error
          
          // Success, decrease lambda
          if lambda > 1e-5 {
            lambda = lambda / 10
          } else {
            break
          }
          inner_success = true
        } else {
          print("[LM INNER] fail, trying to increase lambda")
          // increase lambda and retry
          val = oldval
          if lambda > 1000 {
            print("[LM INNER] giving up in lambda search")
            break
          }
          lambda = lambda * 10
        }
        
        inner_iter_step += 1
        if inner_iter_step > 5 && inner_success {
          break
        }
      }
    }
    print("[FINAL   ] final error = \(gridDataset.graph.error(val))")
    
    let fileManager = FileManager.default
    var filePath = URL(fileURLWithPath: fileManager.currentDirectoryPath)
    filePath.appendPathComponent("./result_pose3.txt")
    
    if fileManager.fileExists(atPath: filePath.path) {
      do {
        try fileManager.removeItem(atPath: filePath.path)
      } catch let error {
        print("error occurred, here are the details:\n \(error)")
      }
    }

    fileManager.createFile(atPath: filePath.path, contents: nil)
    let fileHandle = try! FileHandle(forUpdating: URL(fileURLWithPath: filePath.path))
    var output = FileHandlerOutputStream(fileHandle)
    for i in val.keys.sorted() {
      let t = val[i, as: Pose3.self].t
      print("\(t.x), \(t.y), \(t.z)", to: &output)
    }
  }
}
