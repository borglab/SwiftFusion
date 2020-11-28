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

import BeeDataset
import PenguinParallelWithFoundation
import PenguinStructures
import SwiftFusion
import TensorFlow

extension OISTBeeVideo {
  func getUnnormalizedBatch(
    _ appearanceModelSize: (Int, Int), _ batchSize: Int = 200,
    randomFrameCount: Int = 10
  ) -> Tensor<Double> {
    precondition(self.frameIds.count >= randomFrameCount, "requesting too many random frames")
    var deterministicEntropy = ARC4RandomNumberGenerator(seed: 42)
    let frames =
      Dictionary(uniqueKeysWithValues: self.randomFrames(randomFrameCount, using: &deterministicEntropy))
    let labels = frames.values.flatMap { $0.labels }
    let images = labels
      .randomSelectionWithoutReplacement(k: batchSize, using: &deterministicEntropy)
      .map { label in
        frames[label.frameIndex]!.frame.patch(at: label.location, outputSize: appearanceModelSize)
      }

    return Tensor(stacking: images)
  }

  /// Returns a `[N, h, w, c]` batch of normalized patches from bee bounding boxes, and returns the
  /// statistics used to normalize them.
  public func makeBatch(
    appearanceModelSize: (Int, Int),
    randomFrameCount: Int = 10,
    batchSize: Int = 200
  )
    -> (normalized: Tensor<Double>, statistics: FrameStatistics)
  {
    let stacked = getUnnormalizedBatch(appearanceModelSize, batchSize, randomFrameCount: randomFrameCount)
    let statistics = FrameStatistics(stacked)
    return (statistics.normalized(stacked), statistics)
  }

  /// Returns a `[N, h, w, c]` batch of normalized patches from bee bounding boxes, normalized by
  /// `statistics`
  public func makeBatch(
    statistics: FrameStatistics,
    appearanceModelSize: (Int, Int), randomFrameCount: Int = 10,
    batchSize: Int = 200
  ) -> Tensor<Double> {
    let stacked = getUnnormalizedBatch(appearanceModelSize, batchSize, randomFrameCount: randomFrameCount)
    return statistics.normalized(stacked)
  }

  /// Returns a batch of locations of foreground
  /// bee bounding boxes.
  public func makeForegroundBoundingBoxes(
    patchSize: (Int, Int),
    batchSize: Int = 200
  ) -> [(frame: Tensor<Double>?, obb: OrientedBoundingBox)] {
    /// Anything not completely overlapping labels
    var deterministicEntropy = ARC4RandomNumberGenerator(seed: 42)
    let frames = self.randomFrames(self.frames.count, using: &deterministicEntropy)

    // We need `batchSize / frames.count` patches from each frame, plus the remainder of the
    // integer division.
    var patchesPerFrame = Array(repeating: batchSize / frames.count, count: frames.count)
    patchesPerFrame[0] += batchSize % frames.count

    let obbs = zip(patchesPerFrame, frames).flatMap { args -> [(frame: Tensor<Double>?, obb: OrientedBoundingBox)] in
      let (patchCount, (_, (frame, labels))) = args
      let locations = labels.randomSelectionWithoutReplacement(k: patchCount, using: &deterministicEntropy).map(\.location.center)
      return locations.map { location -> (frame: Tensor<Double>?, obb: OrientedBoundingBox) in
        return (frame: frame, obb: OrientedBoundingBox(
            center: location,
            rows: patchSize.0, cols: patchSize.1))
      }
    }

    return obbs
  }
  
  /// Returns a batch of locations of background
  /// bee bounding boxes.
  public func makeBackgroundBoundingBoxes(
    patchSize: (Int, Int),
    batchSize: Int = 200
  ) -> [(frame: Tensor<Double>?, obb: OrientedBoundingBox)] {
    /// Anything not completely overlapping labels
    let maxSide = min(patchSize.0, patchSize.1)

    var deterministicEntropy = ARC4RandomNumberGenerator(seed: 42)
    let frames = self.randomFrames(self.frames.count, using: &deterministicEntropy)

    // We need `batchSize / frames.count` patches from each frame, plus the remainder of the
    // integer division.
    var patchesPerFrame = Array(repeating: batchSize / frames.count, count: frames.count)
    patchesPerFrame[0] += batchSize % frames.count

    let obbs = zip(patchesPerFrame, frames).flatMap { args -> [(frame: Tensor<Double>?, obb: OrientedBoundingBox)] in
      let (patchCount, (_, (frame, labels))) = args
      let locations = (0..<patchCount).map { _ -> Vector2 in
        let attemptCount = 1000
        for _ in 0..<attemptCount {
          // Sample a point uniformly at random in the frame, away from the edges.
          let location = Vector2(
            Double.random(in: Double(maxSide)..<Double(frame.shape[1] - maxSide), using: &deterministicEntropy),
            Double.random(in: Double(maxSide)..<Double(frame.shape[0] - maxSide), using: &deterministicEntropy))

          // Conservatively reject any point that could possibly overlap with any of the labels.
          for label in labels {
            if (label.location.center.t - location).norm < Double(maxSide) {
              continue
            }
          }

          // The point was not rejected, so return it.
          return location
        }
        fatalError("could not find backround location after \(attemptCount) attempts")
      }
      return locations.map { location -> (frame: Tensor<Double>?, obb: OrientedBoundingBox) in
        let theta = Double.random(in: 0..<(2 * .pi), using: &deterministicEntropy)
        return (frame: frame, obb: OrientedBoundingBox(
            center: Pose2(Rot2(theta), location),
            rows: patchSize.0, cols: patchSize.1))
      }
    }

    return obbs
  }

  /// Returns a `[N, h, w, c]` batch of normalized patches that do not overlap with any bee
  /// bee bounding boxes.
  public func makeBackgroundBatch(
    patchSize: (Int, Int), appearanceModelSize: (Int, Int),
    statistics: FrameStatistics,
    randomFrameCount: Int = 10,
    batchSize: Int = 200
  ) -> Tensor<Double> {
    /// Anything not completely overlapping labels
    let maxSide = min(patchSize.0, patchSize.1)

    var deterministicEntropy = ARC4RandomNumberGenerator(seed: 42)
    let frames = self.randomFrames(randomFrameCount, using: &deterministicEntropy)

    // We need `batchSize / frames.count` patches from each frame, plus the remainder of the
    // integer division.
    var patchesPerFrame = Array(repeating: batchSize / frames.count, count: frames.count)
    patchesPerFrame[0] += batchSize % frames.count

    let images = zip(patchesPerFrame, frames).flatMap { args -> [Tensor<Double>] in
      let (patchCount, (_, (frame, labels))) = args
      let locations = (0..<patchCount).map { _ -> Vector2 in
        let attemptCount = 1000
        for _ in 0..<attemptCount {
          // Sample a point uniformly at random in the frame, away from the edges.
          let location = Vector2(
            Double.random(in: Double(maxSide)..<Double(frame.shape[1] - maxSide), using: &deterministicEntropy),
            Double.random(in: Double(maxSide)..<Double(frame.shape[0] - maxSide), using: &deterministicEntropy))

          // Conservatively reject any point that could possibly overlap with any of the labels.
          for label in labels {
            if (label.location.center.t - location).norm < Double(maxSide) {
              continue
            }
          }

          // The point was not rejected, so return it.
          return location
        }
        fatalError("could not find backround location after \(attemptCount) attempts")
      }
      return locations.map { location -> Tensor<Double> in
        let theta = Double.random(in: 0..<(2 * .pi), using: &deterministicEntropy)
        return frame.patch(
          at: OrientedBoundingBox(
            center: Pose2(Rot2(theta), location),
            rows: patchSize.0, cols: patchSize.1),
          outputSize: appearanceModelSize)
      }
    }

    let stacked = Tensor(stacking: images)
    return statistics.normalized(stacked)
  }

  typealias LabeledFrame = (frameId: Int, (frame: Tensor<Double>, labels: [OISTBeeLabel]))
  
  /// Returns `count` random frames.
  private func randomFrames<R: RandomNumberGenerator>(_ count: Int, using randomness: inout R)
  -> [LabeledFrame]
  {
    let selectedFrameIndices =
      self.frameIds.indices.randomSelectionWithoutReplacement(k: count, using: &randomness)
    return [LabeledFrame](
      unsafeUninitializedCapacity: count
    ) {
      (b, actualCount) -> Void in
      ComputeThreadPools.local.parallelFor(n: count) { (i, _) -> Void in
        let frameIndex = selectedFrameIndices[i]
        let frameId = self.frameIds[frameIndex]
        let frame = Tensor<Double>(self.loadFrame(frameId)!)
        let labels = self.labels[frameIndex]
        assert(labels.allSatisfy { $0.frameIndex == frameId })
        (b.baseAddress! + i).initialize(to: (frameId, (frame, labels)))
      }
      actualCount = count
    }
  }
}
