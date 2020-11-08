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
import TensorFlow

/// A video from VOT, with tracking labels.
public struct VOTVideo {
  /// The frames of the video, as tensors of shape `[height, width, channels]`.
  public var frames: [Tensor<Double>]

  /// The ground truth track of the object in this video.
  public var track: [OrientedBoundingBox]
}

extension VOTVideo {
  /// Creates an instance from `videoName` in `votBaseDirectory`.
  ///
  /// Prints an error and fails if files are missing or corrupted.
  ///
  /// To download the dataset, run
  /// "https://github.com/jvlmdr/trackdat/blob/master/scripts/download_vot.sh", e.g.:
  ///   VOT_YEAR=2018 bash download_vot.sh dl/vot2018
  public init?(votBaseDirectory: String, videoName: String) {
    // We'll append frames and track points to these.
    frames = []
    track = []

    // Parse the descriptions and find the description of the requested video.
    let baseURL = URL(fileURLWithPath: votBaseDirectory)
    guard
      let descData = try? Data(contentsOf: baseURL.appendingPathComponent("description.json")),
      let desc = try? JSONSerialization.jsonObject(with: descData)
    else {
      print("description.json not found in \(votBaseDirectory)")
      return nil
    }
    guard
      let sequences = jsonSubscript(desc, "sequences") as? [Any],
      let sequence = sequences.first(where: {
        (jsonSubscript($0, "name") as? String) == videoName
      })
    else {
      print("\(videoName) not found in description.json")
      return nil
    }
    guard
      let sequenceLength = jsonSubscript(sequence, "length") as? Int,
      let channels = jsonSubscript(sequence, "channels"),
      let colorChannel = jsonSubscript(channels, "color"),
      let colorChannelUid = jsonSubscript(colorChannel, "uid") as? String,
      let colorChannelPattern = jsonSubscript(colorChannel, "pattern") as? String
    else {
      print("expected fields not found in \(videoName) in description.json")
      return nil
    }

    // Load the frames.
    let colorChannelZip =
      baseURL.appendingPathComponent("color").appendingPathComponent("\(colorChannelUid).zip")
    let colorChannelDir =
      baseURL.appendingPathComponent("color").appendingPathComponent("\(colorChannelUid)")
    if !FileManager.default.fileExists(atPath: colorChannelDir.path) {
      extractArchive(at: colorChannelZip, to: colorChannelDir)
    }
    for i in 1..<(sequenceLength + 1) {
      let path = colorChannelDir.appendingPathComponent(String(format: colorChannelPattern, i))
      guard FileManager.default.fileExists(atPath: path.path) else {
        print("frame \(path.path) not found")
        return nil
      }
      frames.append(Tensor<Double>(Image(contentsOf: path).tensor))
    }

    // Load the track points.
    let annotationsZip =
      baseURL.appendingPathComponent("annotations").appendingPathComponent("\(videoName).zip")
    let annotationsDir =
      baseURL.appendingPathComponent("annotations").appendingPathComponent("\(videoName)")
    let trackPath = annotationsDir.appendingPathComponent("groundtruth.txt")
    if !FileManager.default.fileExists(atPath: annotationsDir.path) {
      extractArchive(at: annotationsZip, to: annotationsDir)
    }
    guard let trackString = try? String(contentsOf: trackPath) else {
      print("\(trackPath) not found")
      return nil
    }
    let trackLines = trackString.split(separator: "\n")
    for (i, line) in trackLines.enumerated() {
      let coordinates = line.split(separator: ",").map { Double($0)! }
      let vertices = (0..<4).map { i in Vector2(coordinates[2 * i], coordinates[2 * i + 1]) }

      // Check that the vertices form a rectangle, by checking that opposing sides have matching
      // lengths and opposite directions.
      let sides = (0..<4).map { i in vertices[i] - vertices[(i + 1) % 4] }
      func sidesAgree(_ s1: Vector2, _ s2: Vector2) -> Bool {
        return (s1 + s2).norm < 1
      }
      guard sidesAgree(sides[0], sides[2]) && sidesAgree(sides[1], sides[3]) else {
        print("Region \(i) is not rectangular")
        return nil
      }

      let center = 0.25 * vertices.reduce(.zero, +)
      let width = Int(sides[0].norm)
      let height = Int(sides[1].norm)
      let rot = Rot2(direction: 0.5 * (vertices[1] + vertices[2]) - center)
      track.append(OrientedBoundingBox(center: Pose2(rot, center), rows: height, cols: width))
    }
  }
}

fileprivate func jsonSubscript(_ object: Any, _ key: String) -> Any? {
  guard let dict = object as? [String: Any] else { return nil }
  return dict[key]
}
