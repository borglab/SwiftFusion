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
import PenguinParallelWithFoundation
import PythonKit
import SwiftFusion
import TensorFlow

/// Label type
public enum OISTLabelType {
  /// Full body
  case Body

  /// Only the butt
  case Butt
}

/// A video of bees, with only labels.
public struct OISTBeeVideo {
  /// Frame IDs sorted
  public var frameIds: [Int]

  /// The frames of the video, as tensors of shape `[height, width, channels]`.
  public var frames: [Tensor<Float>]

  /// Labels for this video.
  public var labels: [[OISTBeeLabel]]

  /// Tracks for this video.
  public var tracks: [OISTBeeTrack]

  var directory: URL?

  /// FPS of the video sequence
  public let fps: Int

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

  public init(frameIndex: Int,
    label: OISTLabelType,
    rawLocation: (Double, Double, Double),
    offset: (Int, Int),
    scale: Double
  ) {
    self.frameIndex = frameIndex
    self.label = label
    self.rawLocation = rawLocation
    self.offset = offset
    self.scale = scale
  }
}

extension OISTBeeLabel {
  public func toString() -> String {
    "\(offset.0)\t\(offset.1)\t\((label == .Body) ? 1 : 2)\t\(rawLocation.0)\t\(rawLocation.1)\t\(rawLocation.2)"
  }
}

/// A sequence of bounding boxes tracking a single bee in a video.
public struct OISTBeeTrack {
  /// The frame within the video where this track starts.
  public var startFrameIndex: Int

  /// The positions of the bee at each frame.
  public var boxes: [OrientedBoundingBox]
}

/// For output integers with padding
fileprivate extension String.StringInterpolation {
    mutating func appendInterpolation(_ val: Int, padding: UInt) {
        appendLiteral(String(format: "%0\(padding)d", arguments: [val]))
    }
}

/// For matching with multiple ranges
fileprivate extension String {    
    func ranges(of substring: String, options: CompareOptions = [], locale: Locale? = nil) -> [Range<Index>] {
        var ranges: [Range<Index>] = []
        while let range = range(of: substring, options: options, range: (ranges.last?.upperBound ?? self.startIndex)..<self.endIndex, locale: locale) {
            ranges.append(range)
        }
        return ranges
    }
}

/// Make a filename according to the naming convention
public func formOISTFilename(_ fps: Int, _ index: Int) -> String {
  "frame_\(fps)fps_\(index, padding: 6).txt"
}

/// Loads labels for one frame
func loadFrameLabels(directory: URL, _ index: Int, fps: Int) -> [OISTBeeLabel]? {
  let path = directory.appendingPathComponent("frames_txt").appendingPathComponent(formOISTFilename(fps, index))
  guard let track = try? String(contentsOf: path) else { return nil }
  let lines = track.split(separator: "\n")
  
  return try! lines.lazy.enumerated().map { (id, line) in
    let split = line.split(separator: "\t")
    var labelType: OISTLabelType = .Body
    assert(split.count == 6)
    switch Int(split[2])! {
      case 1: labelType = .Body
      case 2: labelType = .Butt
      default: throw OISTBeeVideo.ParseError(lineIndex: id, line: String(line), message: "Bad label ID!")
    }
  
    return OISTBeeLabel(
      frameIndex: index,
      label: labelType,
      rawLocation: (Double(split[3])!, Double(split[4])!, Double(split[5])!),
      offset: (Int(split[0])!, Int(split[1])!),
      scale: Double(2.0)
    )
  }
}

/// Loads labels for all frames specified by `frameIds` as a collection
func getAllLabels(directory: URL, forFrames frameIds: [Int], fps: Int) -> [[OISTBeeLabel]] {
  var labels: [[OISTBeeLabel]] = []
  while labels.count < frameIds.count,
    let label = loadFrameLabels(directory: directory, frameIds[labels.count], fps: fps)
  {
    labels.append(label)
    if frameIds.count == labels.count {
      break
    }
  }

  return labels
}

extension OISTBeeVideo {
  /// Creates an instance with auto download.
  ///
  /// Parameter frameCount: If specified, truncates the full dataset to this number of frames.
  public init?(deferLoadingFrames: Bool = false, truncate frameCount: Int? = nil) {
    let dataset = Self.downloadDatasetIfNotPresent()
    self.init(directory: dataset, deferLoadingFrames: deferLoadingFrames, length: frameCount)
  }

  /// Creates an instance with auto download.
  ///
  /// Parameter length: If specified, truncates the full dataset to this number of frames.
  public init?(deferLoadingFrames: Bool = false, afterIndex i: Int = 0, length n: Int? = nil) {
    let dataset = Self.downloadDatasetIfNotPresent()
    self.init(
      directory: dataset, deferLoadingFrames: deferLoadingFrames, afterIndex: i,
      length: n
    )
  }

  /// Creates an instance from the data in the given `directory`.
  ///
  /// The directory must contain "frames" and "frames_txt". If the directory contains "tracks",
  /// then tracks will be loaded from that directory.
  ///
  /// The format of a track file is:
  /// - Arbitrarily many lines "offset_x, offset_y, bee_type, x, y, angle".
  ///
  /// Parameter frameCount: If specified, truncates the full dataset to this number of frames.
  public init?(
    directory: URL, deferLoadingFrames: Bool = true, fps: Int = 30, afterIndex i0: Int = 0, length frameCount: Int? = nil
  ) {
    self.frames = []
    self.tracks = []
    self.labels = []
    self.directory = directory
    self.fps = fps

    let directoryContents = try! FileManager.default.contentsOfDirectory(
      at: directory.appendingPathComponent("frames"), includingPropertiesForKeys: nil,
      options: [.skipsHiddenFiles, .skipsPackageDescendants]
    )

    /// This extracts the image sequence number from the file names
    /// Format: `frame_<fps>fps_<id>.png`
    let originalFrameIds: [Int] = directoryContents.map { (f: URL) -> Int in
      let name = f.deletingPathExtension().lastPathComponent
      let ranges = name.ranges(of: #"([0-9]+)"#, options: .regularExpression)
      if ranges.count == 2 {
        return Int(name[ranges[1]])!
      } else {
        fatalError("Bad file name!")
      }
    }.sorted()

    if let frameCount = frameCount {
      self.frameIds = Array(originalFrameIds[i0..<i0 + frameCount])
    } else {
      self.frameIds = originalFrameIds
    }

    // Lazy loading
    if !deferLoadingFrames {
      self.frames = Array(unsafeUninitializedCapacity: self.frameIds.count) {
        (b, actualCount) -> Void in
        ComputeThreadPools.local.parallelFor(n: self.frameIds.count) { (i, _) in
          (b.baseAddress! + i).initialize(to: loadFrame(self.frameIds[i])!)
        }
        actualCount = self.frameIds.count
      }
    }

    self.labels = getAllLabels(directory: directory, forFrames: self.frameIds, fps: fps)

    func loadTrack(_ path: URL) -> OISTBeeTrack {
      let track = try! String(contentsOf: path)
      let lines = track.split(separator: "\n")
      let startFrame = Int(lines.first!)!
      let boxes = lines
        .dropFirst()
        .prefix(max(0, self.frameIds.count - startFrame))
        .map { line -> OrientedBoundingBox in
          let split = line.split(separator: " ")
          return OrientedBoundingBox(
            center: Pose2(
              Rot2(Double(split[2])!),
              Vector2(Double(split[0])!, Double(split[1])!)),
            rows: Int(split[3])!,
            cols: Int(split[4])!)
        }
      return OISTBeeTrack(startFrameIndex: startFrame, boxes: boxes)
    }

    if let trackDirContents = try? FileManager.default.contentsOfDirectory(
      at: directory.appendingPathComponent("tracks"),
      includingPropertiesForKeys: nil
    ) {
      self.tracks = trackDirContents
        .sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
        .map(loadTrack)
        .filter { $0.boxes.count >= min(10, self.frameIds.count) }
    } else {
      print("WARNING: No ground truth tracks found.")
    }
  }

  /// Loads one image frame and downsample by `scale`
  /// Example: frame_30fps_003525.txt
  public func loadFrame(_ index: Int) -> Tensor<Float>? {
    let path = self.directory!.appendingPathComponent("frames").appendingPathComponent("frame_\(fps)fps_\(index, padding: 6).png")
    guard FileManager.default.fileExists(atPath: path.path) else { return nil }
    let downsampler = AvgPool2D<Float>(poolSize: (scale, scale), strides: (scale, scale), padding: .same)
    
    let dots = TensorRange.ellipsis
    let grayscale = ModelSupport.Image(contentsOf: path).tensor.expandingShape(at: 0)[dots, 0].expandingShape(at: 3)

    return downsampler(grayscale).squeezingShape(at: 0)
  }
  
  private static func downloadDatasetIfNotPresent() -> URL {
    let downloadDir = DatasetUtilities.defaultDirectory.appendingPathComponent(
      "oist_bee_videos", isDirectory: true)
    let directoryExists = FileManager.default.fileExists(atPath: downloadDir.path)
    let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadDir.path)
    let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

    guard !directoryExists || directoryEmpty else { return downloadDir }

    let remoteRoot = URL(
      string: "https://storage.googleapis.com/swift-tensorflow-misc-files/beetracking/oist-dataset")!

    let _ = DatasetUtilities.downloadResource(
      filename: "frame_imgs_30fps", fileExtension: "tgz",
      remoteRoot: remoteRoot, localStorageDirectory: downloadDir
    )
    let _ = DatasetUtilities.downloadResource(
      filename: "frames_txt", fileExtension: "zip",
      remoteRoot: remoteRoot, localStorageDirectory: downloadDir
    )
    let _ = DatasetUtilities.downloadResource(
      filename: "tracks", fileExtension: "tar.gz",
      remoteRoot: remoteRoot, localStorageDirectory: downloadDir
    )

    return downloadDir
  }
}

extension OISTBeeTrack {
  /// Writes an animation of the track to files "track000.png", "track001.png", ... in `directory`.
  ///
  /// Parameter video: The video containing the track.
  /// Parameter cameraSize: The size of the output animation. The rendered video is a subregion of
  /// the source video, so that it is easier to see the details relevant to the track.
  public func render(to directory: String, video: OISTBeeVideo, cameraSize: Int = 200) {
    let plt = Python.import("matplotlib.pyplot")

    try! FileManager.default.createDirectory(
      atPath: directory,
      withIntermediateDirectories: true)

    // Exponential average of the bounding box center, so that the "camera" can smoothly follow
    // the track, which makes it much easier to tell when mistakes happen.
    var currentCenter = self.boxes[0].center.t

    for frameIndex in self.startFrameIndex..<(self.startFrameIndex + self.boxes.count) {
      let box = self.boxes[frameIndex - self.startFrameIndex]

      // Update the exponential average.
      currentCenter = 0.9 * currentCenter + 0.1 * box.center.t

      // Calculate the bounding box corners relative to the current camera position.
      let surroundingBox = OrientedBoundingBox(
        center: Pose2(Rot2(), currentCenter), rows: cameraSize, cols: cameraSize)
      let corners = box.corners.map {
        surroundingBox.center.inverse() * $0
          + Vector2(Double(cameraSize) / 2, Double(cameraSize) / 2)
      }

      // Plot the frame.
      let frame = ArrayImage(video.loadFrame(video.frameIds[frameIndex])!).patch(at: surroundingBox).tensor
      let figure = plt.figure(figsize: [20, 20])
      let subplot = figure.add_subplot(1, 1, 1)
      subplot.imshow(frame.makeNumpyArray() / 255.0)
      subplot.plot(corners.map(\.x), corners.map(\.y))
      subplot.text(10, 10, "frame \(frameIndex)", color: "yellow")
      figure.savefig(directory + "/" + String(format: "frame%03d.png", frameIndex))
      plt.close(figure)
    }
  }
}
