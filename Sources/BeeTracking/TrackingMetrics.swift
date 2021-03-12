import BeeDataset
import Foundation
import PenguinParallelWithFoundation
import PythonKit
import SwiftFusion
import TensorFlow

/// Accuracy and robustness for a subsequence, as defined in the VOT 2020 challenge [1].
///
/// [1] http://prints.vicos.si/publications/384
public struct SubsequenceMetrics: Codable {
  /// Accuracy as defined in [1], equation (1).
  public let accuracy: Double

  /// Robustness as defined in [1], equation (2).
  public let robustness: Double

  /// The index of the first tracking failure in this subsequence. This is N^F_{s,a} from [1].
  ///
  /// If there was not a tracking failure in this subsequence, then this is the length of the
  /// subsequence.
  public let NFsa: Int

  /// Number of frames in this subsequence. This is N_{s, a} from [1].
  public let Nsa: Int

  /// averageOverlap[i] is the average overlap over all frames from 0 to i, inclusive.
  ///
  /// This is only defined up to the failure frame, i.e. `averageOverlap.count == NFsa`.
  private let averageOverlap: [Double]

  /// overlap[i] is the raw overlap over all frames from 0 to i, inclusive.
  ///
  /// This is defined on all frames.
  public let overlap: [Double]

  public init(groundTruth: [OrientedBoundingBox], prediction: [OrientedBoundingBox]) {
    precondition(groundTruth.count == prediction.count)

    let overlaps = zip(groundTruth, prediction).map { $0.0.overlap($0.1) }

    // Find the first failure frame.
    var NFsa = prediction.count
    for (index, overlap) in overlaps.enumerated() {
      if overlap < 0.1 {
        NFsa = index
        break
      }
    }

    // Compute the average overlaps.
    var averageOverlap = [Double]()
    averageOverlap.reserveCapacity(NFsa)
    averageOverlap.append(overlaps[0])
    for i in 1..<NFsa {
      averageOverlap.append((Double(i) * averageOverlap.last! + overlaps[i]) / Double(i + 1))
    }

    self.averageOverlap = averageOverlap
    self.NFsa = NFsa
    self.Nsa = overlaps.count
    self.accuracy = overlaps[0..<NFsa].reduce(0, +) / max(1, Double(NFsa))
    self.robustness = Double(NFsa) / max(1, Double(Nsa))
    self.overlap = overlaps
  }

  public init(
    accuracy: Double,
    robustness: Double,
    NFsa: Int,
    Nsa: Int,
    averageOverlap: [Double],
    overlap: [Double] = []
  ) {
    self.accuracy = accuracy
    self.robustness = robustness
    self.NFsa = NFsa
    self.Nsa = Nsa
    self.averageOverlap = averageOverlap
    self.overlap = overlap
  }

  /// The extended average overlap. This is \phi_{s,a}(i) from [1].
  ///
  /// Note that if tracking never failed in this subsequence, then we cannot extend the average
  /// overlap past the end of the sequence.
  public func extendedAverageOverlap(_ i: Int) -> Double? {
    if i < averageOverlap.count { return averageOverlap[i] }
    if NFsa == Nsa { return nil }
    return 0
  }
}

/// Accuracy and robustness for a sequence, as defined in [1].
public struct SequenceMetrics: Codable {
  /// Accuracy as defined in [1], equation (3).
  public let accuracy: Double

  /// Robustness as defined in [1], equation (4).
  public let robustness: Double

  /// The number of frames used to calculate accuracy in this sequence. This is N^F_s from [1].
  public let NFs: Int

  /// The number of frames in the sequence. This is N_s from [1].
  public let Ns: Int

  public init(_ subsequences: [SubsequenceMetrics]) {
    self.NFs = subsequences.map { $0.NFsa }.reduce(0, +)
    self.Ns = subsequences.map { $0.Nsa }.max()!
    self.accuracy = subsequences.map { $0.accuracy * Double($0.NFsa) }.reduce(0, +) / max(1, Double(self.NFs))
    self.robustness = subsequences.map { $0.robustness * Double($0.Nsa) }.reduce(0, +)
      / max(1, Double(subsequences.map { $0.Nsa }.reduce(0, +)))
  }

  public init(
    accuracy: Double,
    robustness: Double,
    NFs: Int,
    Ns: Int,
    averageOverlap: [Double]
  ) {
    self.accuracy = accuracy
    self.robustness = robustness
    self.NFs = NFs
    self.Ns = Ns
  }
}

/// Accuracy and robustness for a tracker, as defined in [1].
public struct TrackerMetrics: Codable {
  /// Accuracy as defined in [1], equation (5).
  public let accuracy: Double

  /// Robustness as defined in [1], equation (6).
  public let robustness: Double

  public init(_ sequences: [SequenceMetrics]) {
    self.accuracy = sequences.map { $0.accuracy * Double($0.NFs) }.reduce(0, +)
      / max(1, Double(sequences.map { $0.NFs }.reduce(0, +)))
    self.robustness = sequences.map { $0.robustness * Double($0.Ns) }.reduce(0, +)
      / max(1, Double(sequences.map { $0.Ns }.reduce(0, +)))
  }
}

/// Expected Averge Overlap (EAO) for a tracker, as defined in [1].
public struct ExpectedAverageOverlap: Codable {
  /// The values in the EAO curve.
  public let curve: [Double]

  public init(_ subsequences: [SubsequenceMetrics]) {
    let count = subsequences.map { $0.Nsa }.max()!
    var curve = [Double]()
    curve.reserveCapacity(count)
    for i in 0..<count {
      var availableSubsequenceCount = 0
      var totalAverageOverlap: Double = 0
      for subsequence in subsequences {
        if let averageOverlap = subsequence.extendedAverageOverlap(i) {
          availableSubsequenceCount += 1
          totalAverageOverlap += averageOverlap
        }
      }
      curve.append(totalAverageOverlap / max(1, Double(availableSubsequenceCount)))
    }
    self.curve = curve
  }
}

/// A dataset on which to evaluate trackers.
public struct TrackerEvaluationDataset {
  /// The videos and ground truth labels in the dataset.
  public let sequences: [TrackerEvaluationSequence]
}

extension TrackerEvaluationDataset {
  /// Evaluate the performance of `tracker` on `self`.
  ///
  /// Initializes the tracker at multiple points in each sequence, `0`, `deltaAnchor`, `2 * deltaAnchor`, etc,
  /// and computes metrics at each subsequence.
  ///
  /// Writes metrics for each sequence `i` to `"\(outputFile)-sequence\(i).json"` and writes combined metrics
  /// for all sequences to `"\(outputFile).json"`.
  ///
  /// Parameter sequenceCount: How many sequences from `self` to use during the evaluation.
  public func evaluate(
    _ tracker: Tracker,
    sequenceCount: Int,
    deltaAnchor: Int,
    outputFile: String
  ) -> TrackerEvaluationResults {
    let sequenceEvaluations = sequences.prefix(sequenceCount).enumerated().map {
      (i, sequence) -> SequenceEvaluationResults in
      print("Evaluating sequence \(i + 1) of \(sequenceCount)")
      return sequence.evaluate(tracker, deltaAnchor: deltaAnchor, outputFile: "\(outputFile)-sequence\(i)")
    }
    let result = TrackerEvaluationResults(
      sequences: sequenceEvaluations,
      trackerMetrics: TrackerMetrics(sequenceEvaluations.map { $0.sequenceMetrics }),
      expectedAverageOverlap: ExpectedAverageOverlap(
        sequenceEvaluations.flatMap { $0.subsequences }.map { $0.metrics }))

    return result
  }
}

/// A single sequence in a `TrackerEvaluationDataset`.
public struct TrackerEvaluationSequence {
  /// The frames of the sequence.
  public let frames: [Tensor<Float>]

  /// The ground truth labels for the sequence.
  public let groundTruth: [OrientedBoundingBox]

  /// Returns subsequences of `self` starting at `0`, `deltaAnchor`, 2 * deltaAnchor`, etc.
  func subsequences(deltaAnchor: Int) -> [TrackerEvaluationSequence] {
    stride(from: 0, to: frames.count, by: deltaAnchor).map { startIndex in
      TrackerEvaluationSequence(
        frames: Array(frames[startIndex...]), groundTruth: Array(groundTruth[startIndex...]))
    }
  }
}

extension TrackerEvaluationSequence {
  /// Returns the performance of `tracker` on the sequence `self`.
  ///
  /// Initializes the tracker at multiple points in the sequence, `0`, `deltaAnchor`, `2 * deltaAnchor`, etc,
  /// and returns metrics at each subsequence.
  ///
  /// Writes metrics to `"\(outputFile).json"`.
  public func evaluate(_ tracker: Tracker, deltaAnchor: Int, outputFile: String) -> SequenceEvaluationResults {
    guard let _ = try? Python.attemptImport("shapely") else {
      print("python shapely library must be installed")
      preconditionFailure()
    }

    let subsequences = self.subsequences(deltaAnchor: deltaAnchor)
    let subsequencePredictions = [[OrientedBoundingBox]](
      unsafeUninitializedCapacity: subsequences.count
    ) { (buf, actualCount) in
      // Evaluate up to 4 sequences in parallel.
      // TODO(marcrasi): We should call a high-level "parallel map with limited concurrency method" instead of
      // implementing it ourselves here.
      let blockCount = 4
      ComputeThreadPools.local.parallelFor(n: blockCount) { (blockIndex, _) in
        for i in 0..<subsequences.count {
          guard i % (2 * blockCount) == blockIndex
            || i % (2 * blockCount) == 2 * blockCount - 1 - blockIndex
          else {
            continue
          }
          let subsequence = subsequences[i]
          print("Evaluating subsequence \(i + 1) of \(subsequences.count)")
          (buf.baseAddress! + i).initialize(to: tracker(subsequence.frames, subsequence.groundTruth[0]))
        }
      }
      actualCount = subsequences.count
    }
    let subsequenceEvaluations = zip(subsequences, subsequencePredictions).map {
      SubsequenceEvaluationResults(
        metrics: SubsequenceMetrics(groundTruth: $0.0.groundTruth, prediction: $0.1),
        prediction: $0.1,
        groundTruth: $0.0.groundTruth,
        frames: $0.0.frames)
    }

    let result = SequenceEvaluationResults(
      subsequences: subsequenceEvaluations,
      sequenceMetrics: SequenceMetrics(subsequenceEvaluations.map { $0.metrics }))

    return result
  }
}

extension TrackerEvaluationDataset {
  /// Creates an instance using a video from OIST.
  public init(_ video: OISTBeeVideo) {
    precondition(video.frames.count == video.frameIds.count, "all the frames must be loaded")
    var sequences = [TrackerEvaluationSequence]()
    sequences.reserveCapacity(video.tracks.count)
    for track in video.tracks {
      let sequence = TrackerEvaluationSequence(
        frames: Array(
          video.frames[track.startFrameIndex..<(track.boxes.count)]),
        groundTruth: track.boxes)
      sequences.append(sequence)
    }
    self.sequences = sequences
  }
}

/// All the tracker evaluation metrics in one struct.
public struct TrackerEvaluationResults: Codable {
  /// The sequence results for all the sequences in the dataset.
  public let sequences: [SequenceEvaluationResults]

  /// The overall tracker metrics on this dataset.
  public let trackerMetrics: TrackerMetrics

  /// The overall expected average overlap curve for the tracker on this dataset.
  public let expectedAverageOverlap: ExpectedAverageOverlap
}

/// All the sequence evaluation metrics in one struct.
public struct SequenceEvaluationResults: Codable {
  /// The subsequence metrics for all subsequences in this sequence. And the predictions.
  public let subsequences: [SubsequenceEvaluationResults]

  /// The sequence metrics for this sequence.
  public let sequenceMetrics: SequenceMetrics
}

public struct SubsequenceEvaluationResults: Codable {
  public let metrics: SubsequenceMetrics
  public let prediction: [OrientedBoundingBox]
  public let groundTruth: [OrientedBoundingBox]
  public let frames: [Tensor<Float>]
}

/// Given `frames` and a `start` region containing an object to track, returns predicted regions
/// for all `frames` (including the first one).
public typealias Tracker =
  (_ frames: [Tensor<Float>], _ start: OrientedBoundingBox) -> [OrientedBoundingBox]
