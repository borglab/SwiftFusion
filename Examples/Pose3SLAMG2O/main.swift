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
/// Usage: Pose3SLAMG2O [path to .g2o file]
///
/// Missing features:
/// - Does not take g2o information matrix into account.
/// - Does not use a proper general purpose solver.
/// - Has not been compared against other implementations, so it could be wrong.

import Foundation
import SwiftFusion
import TensorFlow

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

func main() {
  // Parse commandline.
  guard CommandLine.arguments.count == 3 else {
    print("Usage: Pose3SLAMG2O [path to .g2o file] [path to output csv file]")
    return
  }
  let g2oURL = URL(fileURLWithPath: CommandLine.arguments[1])
  let fileManager = FileManager.default
  let filePath = URL(fileURLWithPath: CommandLine.arguments[2])
  
  print("Storing result at \(filePath.path)")
  
  if fileManager.fileExists(atPath: filePath.path) {
    do {
      try fileManager.removeItem(atPath: filePath.path)
    } catch let error {
      print("error occurred, here are the details:\n \(error)")
    }
  }
  
  // Load .g2o file.
  let problem = try! G2OReader.G2ONewFactorGraph(g2oFile3D: g2oURL)

  var val = problem.initialGuess
  var graph = problem.graph
  
  graph.store(NewPriorFactor3(TypedID(0), Pose3(Rot3.fromTangent(Vector3.zero), Vector3.zero)))
  
  var optimizer = LM()
  optimizer.verbosity = .TRYLAMBDA
  
  do {
    try optimizer.optimize(graph: graph, initial: &val)
  } catch let error {
    print("The solver gave up, message: \(error.localizedDescription)")
  }
  
  fileManager.createFile(atPath: filePath.path, contents: nil)
  let fileHandle = try! FileHandle(forUpdating: URL(fileURLWithPath: filePath.path))
  var output = FileHandlerOutputStream(fileHandle)
  
  for i in problem.initialGuessId {
    let t = val[i].t
    output.write("\(t.x), \(t.y), \(t.z)\n")
  }
}

main()
