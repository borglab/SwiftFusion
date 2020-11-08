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

import Datasets
import Foundation
import ModelSupport
import SwiftFusion
import TensorFlow

public enum OISTLabelType {
    case Body
    case Butt
}

// BEE_OBJECT_SIZES = {1: (20, 35),  # bee class is labeled 1
//                     2: (20, 20)}  # butt class is labeled 2

/// A video of bees, with tracking labels.
public struct OISTBeeVideo {
  /// Frame IDs sorted
  public var frameIds: [Int]

  /// The frames of the video, as tensors of shape `[height, width, channels]`.
  public var frames: [Tensor<Double>]

  /// Labels for this video.
  public var labels: [[OISTBeeLabel]]

  var directory: URL?

  /// FPS of the video sequence
  var fps: Int

  /// Scale Factor
  public var scale: Int = 2

  public struct ParseError: Swift.Error {
    public let lineIndex: Int
    public let line: String
    public let message: String
  }
}

/// The location of a bee within a frame of a `OISTBeeVideo`.
public struct OISTBeeLabel {
  /// The frame within the video.
  public var frameIndex: Int

  /// The global location of the bee.
  public var location: OrientedBoundingBox {
    get {
      let bboxSize: (Int, Int) = [
        OISTLabelType.Body: (40, 70),
        OISTLabelType.Butt: (40, 40)
      ][label]!

      return OrientedBoundingBox(
        center: Pose2(
          Rot2(rawLocation.2 - .pi / 2),
          (1/scale) * Vector2(Double(offset.0) + rawLocation.0, Double(offset.1) + rawLocation.1)
        ), rows: bboxSize.0, cols: bboxSize.1
      )
    }
  }

  /// The type of the label
  public var label: OISTLabelType

  /// Raw location read from the data file
  /// (x, y, rotation)
  public var rawLocation: (Double, Double, Double)

  /// Offset of the location
  /// Since the dataset is labeled in a grid-by-grid fashion
  public var offset: (Int, Int)

  /// Scale Factor
  public var scale: Double = 2
}

fileprivate extension String.StringInterpolation {
    mutating func appendInterpolation(_ val: Int, padding: UInt) {
        appendLiteral(String(format: "%0\(padding)d", arguments: [val]))
    }
}

fileprivate extension String {    
    func ranges(of substring: String, options: CompareOptions = [], locale: Locale? = nil) -> [Range<Index>] {
        var ranges: [Range<Index>] = []
        while let range = range(of: substring, options: options, range: (ranges.last?.upperBound ?? self.startIndex)..<self.endIndex, locale: locale) {
            ranges.append(range)
        }
        return ranges
    }
}

extension OISTBeeVideo {
  /// Creates an instance with auto download.
  public init?(deferLoadingFrames: Bool = false) {
    let dataset = Self.downloadDatasetIfNotPresent()
    self.init(directory: dataset, deferLoadingFrames: deferLoadingFrames)
  }

  /// Creates an instance from the data in the given `directory`.
  ///
  /// The directory must contain:
  /// - Frames named "frame0.jpeg", "frame1.jpeg", etc, consecutively.
  /// - Zero or more tracks named "track0.txt", "track1.txt", etc, consecutively.
  ///
  /// The format of a track file is:
  /// - Arbitrarily many lines "offset_x, offset_y, bee_type, x, y, angle".
  public init?(directory: URL, deferLoadingFrames: Bool = true, fps: Int = 30) {
    self.frames = []
    self.labels = []
    self.frameIds = []
    self.directory = directory
    self.fps = fps

    let directoryContents = try! FileManager.default.contentsOfDirectory(at: directory.appendingPathComponent("frames"), includingPropertiesForKeys: nil)
    self.frameIds = directoryContents.map { f in
      let name = f.deletingPathExtension().lastPathComponent
      let ranges = name.ranges(of: #"([0-9]+)"#, options: .regularExpression)
      if ranges.count == 2 {
        return Int(name[ranges[1]])!
      } else {
        fatalError("Bad file name!")
      }
    }.sorted()

    if !deferLoadingFrames {
      while let frame = loadFrame(self.frameIds[self.frames.count]) {
        self.frames.append(frame)
        if self.frameIds.count == self.frames.count {
          break
        }
      }
    }

    func loadTrack(_ index: Int) -> [OISTBeeLabel]? {
      let path = directory.appendingPathComponent("frames_txt").appendingPathComponent("frame_\(fps)fps_\(index, padding: 6).txt")
      guard let track = try? String(contentsOf: path) else { return nil }
      let lines = track.split(separator: "\n")
      
      return try! lines.enumerated().map { (id, line) in
        let split = line.split(separator: "\t")
        var labelType: OISTLabelType = .Body
        assert(split.count == 6)
        switch Int(split[2])! {
          case 1: labelType = .Body
          case 2: labelType = .Butt
          default: throw Self.ParseError(lineIndex: id, line: String(line), message: "Bad label ID!")
        }
      
        return OISTBeeLabel(
          frameIndex: id,
          label: labelType,
          rawLocation: (Double(split[3])!, Double(split[4])!, Double(split[5])!),
          offset: (Int(split[0])!, Int(split[1])!),
          scale: Double(scale)
        )
      }
    }
    while let label = loadTrack(self.frameIds[self.labels.count]) {
      self.labels.append(label)
      if self.frameIds.count == self.labels.count {
          break
        }
    }
  }

  /// Example: frame_30fps_003525.txt
  public func loadFrame(_ index: Int) -> Tensor<Double>? {
    let path = self.directory!.appendingPathComponent("frames").appendingPathComponent("frame_\(fps)fps_\(index, padding: 6).png")
    guard FileManager.default.fileExists(atPath: path.path) else { return nil }
    let downsampler = AvgPool2D<Double>(poolSize: (scale, scale), strides: (scale, scale), padding: .same)
    return downsampler(Tensor<Double>(ModelSupport.Image(contentsOf: path).tensor).expandingShape(at: 0)).squeezingShape(at: 0)
  }
  
  private static func downloadDatasetIfNotPresent() -> URL {
    let downloadDir = DatasetUtilities.defaultDirectory.appendingPathComponent(
      "oist_bee_videos", isDirectory: true)
    let directoryExists = FileManager.default.fileExists(atPath: downloadDir.path)
    let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadDir.path)
    let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

    guard !directoryExists || directoryEmpty else { return downloadDir }

    let remoteRoot = URL(
      string: "https://storage.googleapis.com/swift-tensorflow-misc-files/beetracking")!

    let _ = DatasetUtilities.downloadResource(
      filename: "oist_bee_videos", fileExtension: "tar.gz",
      remoteRoot: remoteRoot, localStorageDirectory: downloadDir
    )

    return downloadDir
  }
}
