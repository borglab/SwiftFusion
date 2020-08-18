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
import SwiftFusion

/// Reads an array of oriented bounding boxes from `file`.
///
/// Each line of the file is an oriented bounding box, in the format:
///   <rotation (radians)> <x> <y>
public func beeOrientedBoundingBoxes(file: URL) -> [OrientedBoundingBox]? {
  // The dimensions of the bee bounding box are hardcoded because they're the same for all the
  // sequences in the dataset.
  let rows = 28
  let cols = 62
  guard let lines = try? String(contentsOf: file) else { return nil }
  return lines.split(separator: "\n").compactMap { line in
    let lineCols = line.split(separator: " ")
    guard lineCols.count == 3,
      let r = Double(lineCols[0]),
      let x = Double(lineCols[1]),
      let y = Double(lineCols[2])
    else { return nil }
    return OrientedBoundingBox(center: Pose2(Rot2(r), Vector2(x, y)), rows: rows, cols: cols)
  }
}

/// Returns an array of oriented bounding boxes that track a bee in the bee sequence named
/// `sequenceName`, automatically downloading the dataset frmo the internet if it is not already
/// present on the local system.
///
/// The dataset is originally frmo https://www.cc.gatech.edu/~borg/ijcv_psslds/, and contains
/// sequences named "seq1", "seq2", ..., "seq6".
///
/// WARNING: The indices of the bounding boxes seem like they are not perfectly aligned with the
/// indices of the `BeeFrames` from the same sequence. (For example, frame `i` might correspond to
/// bounding box `i - 1` or `i - 2`).
public func beeOrientedBoundingBoxes(sequenceName: String) -> [OrientedBoundingBox]? {
  let dir = downloadBeeDatasetIfNotPresent()
  return beeOrientedBoundingBoxes(
    file: dir.appendingPathComponent("obbs").appendingPathComponent("\(sequenceName).txt"))
}
