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

import Foundation
import ModelSupport
import SwiftFusion

/// An ordered collection of frames from a video of bees.
///
/// Note: Use `BeeVideo` instead. This is for backwards compatibility with old code.
public struct BeeFrames: RandomAccessCollection {
  public let directory: URL
  public let frameCount: Int

  /// Creates a `BeeFrames` from the sequence named `sequenceName`, automatically downloading the
  /// dataset from the internet if it is not already present on the local system.
  ///
  /// The dataset is originally frmo https://www.cc.gatech.edu/~borg/ijcv_psslds/, and contains
  /// sequences named "seq1", "seq2", ..., "seq6".
  public init?(sequenceName: String) {
    let dir = downloadBeeDatasetIfNotPresent()
    self.init(directory: dir.appendingPathComponent("frames").appendingPathComponent(sequenceName))
  }

  /// Creates a `BeeFrames` from the data in the given `directory`.
  ///
  /// The directory must contain:
  /// - A file named "index.txt" whose first line is the total number of frames.
  /// - Frames named "frame1.png", "frame2.png", etc.
  public init?(directory: URL) {
    let indexFile = directory.appendingPathComponent("index.txt")
    guard let index = try? String(contentsOf: indexFile) else { return nil }
    guard let indexLine = index.split(separator: "\n").first else {
      fatalError("index.txt empty")
    }
    guard let frameCount = Int(indexLine) else {
      fatalError("index.txt first line is not a number")
    }
    self.directory = directory
    self.frameCount = frameCount
  }

  public var startIndex: Int { 0 }
  public var endIndex: Int { frameCount }

  public func index(before i: Int) -> Int { i - 1 }
  public func index(after i: Int) -> Int { i + 1 }

  public subscript(index: Int) -> Image {
    return Image(jpeg: directory.appendingPathComponent("frame\(index + 1).png"))
  }
}
