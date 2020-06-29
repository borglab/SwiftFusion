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
import TensorBoardX
import TSCUtility

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
  var inputFilename = ""
  var outputFilename = ""
  var loggingFolder: String? = nil
  var useChordal = false
  var useChordalInitialization = false
  // Parse commandline.
  do {
    let parser = ArgumentParser(
          commandName: "Pose3SLAMG2O",
          usage: "[path to .g2o file] [path to output csv file] [path to logging folder]",
          overview: "The command is used for argument parsing",
          seeAlso: "getopt(1)")
    let argsv = Array(CommandLine.arguments.dropFirst())
    let input = parser.add(
          positional: "input",
          kind: String.self,
          usage: "Input g2o file",
          completion: .filename)
    let output = parser.add(
          positional: "output",
          kind: String.self,
          usage: "Output csv file",
          completion: .filename)
    let logging = parser.add(
          option: "--logging",
          shortName: "-l",
          kind: String.self,
          usage: "Tensorboard log folder",
          completion: .filename)
    let chordal = parser.add(
          option: "--chordal",
          shortName: "-c",
          kind: Bool.self,
          usage: "Use Chordal norm in BetweenFactor",
          completion: ShellCompletion.none)
    let chordal_init = parser.add(
          option: "--chordal-init",
          shortName: nil,
          kind: Bool.self,
          usage: "Use Chordal Initialization",
          completion: ShellCompletion.none)
    let parguments = try parser.parse(argsv)
    
    inputFilename = parguments.get(input)!
    outputFilename = parguments.get(output)!
    loggingFolder = parguments.get(logging)
    if let c = parguments.get(chordal) {
      print("Using Chordal norm in BetweenFactor")
      useChordal = c
    }
    if let c = parguments.get(chordal_init) {
      print("Using Chordal Initialization")
      useChordalInitialization = c
    }
  } catch ArgumentParserError.expectedValue(let value) {
    print("Missing value for argument \(value).")
    exit(1)
  } catch ArgumentParserError.expectedArguments(let parser, let stringArray) {
    print("Parser: \(parser) Missing arguments: \(stringArray.joined()).")
    exit(1)
  } catch {
    print(error.localizedDescription)
    exit(1)
  }
  
  let doTracing = loggingFolder != nil
  
  let g2oURL = URL(fileURLWithPath: inputFilename)
  let fileManager = FileManager.default
  let filePath = URL(fileURLWithPath: outputFilename)
  
  print("Storing result at \(filePath.path)")
  
  if fileManager.fileExists(atPath: filePath.path) {
    do {
      try fileManager.removeItem(atPath: filePath.path)
    } catch let error {
      print("error occurred, here are the details:\n \(error)")
    }
  }
  
  var logFileWriter: SummaryWriter? = nil
  let datasetName = g2oURL.deletingPathExtension().lastPathComponent
  
  if doTracing {
    let fileWriterURL = URL(string: loggingFolder!)
    
    if let _f = fileWriterURL {
      logFileWriter = SummaryWriter(logdir: _f, suffix: datasetName)
    }
  }
  
  // Load .g2o file.
  let problem = try! G2OReader.G2OFactorGraph(g2oFile3D: g2oURL, chordal: useChordal)

  var val = problem.initialGuess
  var graph = problem.graph
  
  graph.store(PriorFactor3(TypedID(0), Pose3(Rot3.fromTangent(Vector3.zero), Vector3.zero)))
  
  if useChordalInitialization {
    val = ChordalInitialization.GetInitializations(graph: graph, val: val, ids: problem.variableId)
  }
  
  var optimizer = LM(precision: 1e-1, max_iteration: 100)
  
  optimizer.verbosity = .TRYLAMBDA
  optimizer.max_iteration = 100
  
  do {
    var hook: ((FactorGraph, VariableAssignments, Double, Int) -> Void)? = nil
    if doTracing {
      hook = { fg, val, lambda, step in
        logFileWriter!.addScalar(tag: "optimizer/loss", scalar: fg.error(at: val), globalStep: step)
        logFileWriter!.addScalar(tag: "optimizer/lambda", scalar: lambda, globalStep: step)
        logFileWriter!.flush()
      }
    }
    try optimizer.optimize(graph: graph, initial: &val, hook: hook)
  } catch let error {
    if doTracing {
      logFileWriter!.addText(tag: "optimizer/message", text: "The solver gave up, message: \(error.localizedDescription)")
    }
    print("The solver gave up, message: \(error.localizedDescription)")
  }
  
  fileManager.createFile(atPath: filePath.path, contents: nil)
  let fileHandle = try! FileHandle(forUpdating: URL(fileURLWithPath: filePath.path))
  var output = FileHandlerOutputStream(fileHandle)
  
  for i in problem.variableId {
    let t = val[i].t
    output.write("\(t.x), \(t.y), \(t.z)\n")
  }
}

main()
