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
import Foundation
import ModelSupport
import SwiftFusion
import TensorFlow

/// A video of bees, with tracking labels.
public struct BeeVideo {
  /// The frames of the video, as tensors of shape `[height, width, channels]`.
  public var frames: [Tensor<Double>]

  /// Labeled tracks for this video.
  public var tracks: [[BeeTrackPoint]]

  var directory: URL?
}

/// The location of a bee within a frame of a `BeeVideo`.
public struct BeeTrackPoint {
  /// The frame within the video.
  public var frameIndex: Int

  /// The location of the bee.
  public var location: OrientedBoundingBox
}

extension BeeVideo {
  /// Creates an instance from the video named `videoName`.
  public init?(videoName: String, deferLoadingFrames: Bool = false) {
    let dataset = Self.downloadDatasetIfNotPresent()
    self.init(directory: dataset.appendingPathComponent(videoName), deferLoadingFrames: deferLoadingFrames)
  }

  /// Creates an instance from the data in the given `directory`.
  ///
  /// The directory must contain:
  /// - Frames named "frame0.jpeg", "frame1.jpeg", etc, consecutively.
  /// - Zero or more tracks named "track0.txt", "track1.txt", etc, consecutively.
  ///
  /// The format of a track file is:
  /// - A line "<height> <width>" specifying the size of the bounding boxes in the track.
  /// - Followed by arbitrarily many lines "<frame index> <x> <y> <theta>".
  public init?(directory: URL, deferLoadingFrames: Bool = false) {
    self.frames = []
    self.tracks = []
    self.directory = directory

    while let frame = loadFrame(self.frames.count) {
      self.frames.append(frame)
    }

    func loadTrack(_ index: Int) -> [BeeTrackPoint]? {
      let path = directory.appendingPathComponent("track\(index).txt")
      guard let track = try? String(contentsOf: path) else { return nil }
      let lines = track.split(separator: "\n")
      let bbSize = lines.first!.split(separator: " ")
      let bbHeight = Int(bbSize[0])!
      let bbWidth = Int(bbSize[1])!
      return lines.dropFirst().map { line in
        let split = line.split(separator: " ")
        return BeeTrackPoint(
          frameIndex: Int(split[0])!,
          location: OrientedBoundingBox(
            center: Pose2(
              Rot2(Double(split[3])!),
              Vector2(Double(split[1])!, Double(split[2])!)),
            rows: bbHeight,
            cols: bbWidth))
      }
    }
    while let track = loadTrack(self.tracks.count) {
      self.tracks.append(track)
    }
  }

  public func loadFrame(_ index: Int) -> Tensor<Double>? {
    let path = self.directory!.appendingPathComponent("frame\(index).jpeg")
    guard FileManager.default.fileExists(atPath: path.path) else { return nil }
    return Tensor<Double>(Image(contentsOf: path).tensor)
  }
  
  private static func downloadDatasetIfNotPresent() -> URL {
    let downloadDir = DatasetUtilities.defaultDirectory.appendingPathComponent(
      "bee_videos", isDirectory: true)
    let directoryExists = FileManager.default.fileExists(atPath: downloadDir.path)
    let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadDir.path)
    let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

    guard !directoryExists || directoryEmpty else { return downloadDir }

    let remoteRoot = URL(
      string: "https://storage.googleapis.com/swift-tensorflow-misc-files/beetracking")!

    let _ = DatasetUtilities.downloadResource(
      filename: "bee_videos", fileExtension: "tar.gz",
      remoteRoot: remoteRoot, localStorageDirectory: downloadDir
    )

    return downloadDir
  }
}
