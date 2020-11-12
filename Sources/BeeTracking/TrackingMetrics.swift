import BeeDataset
import PythonKit
import SwiftFusion
import TensorFlow

/// Accuracy and robustness for a subsequence, as defined in the VOT 2020 challenge [1].
///
/// [1] http://prints.vicos.si/publications/384
public struct SubsequenceMetrics {
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
    self.accuracy = overlaps[0..<NFsa].reduce(0, +) / Double(NFsa)
    self.robustness = Double(NFsa) / Double(Nsa)
  }

  public init(
    accuracy: Double,
    robustness: Double,
    NFsa: Int,
    Nsa: Int,
    averageOverlap: [Double]
  ) {
    self.accuracy = accuracy
    self.robustness = robustness
    self.NFsa = NFsa
    self.Nsa = Nsa
    self.averageOverlap = averageOverlap
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
public struct SequenceMetrics {
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
    self.accuracy = subsequences.map { $0.accuracy * Double($0.NFsa) }.reduce(0, +) / Double(self.NFs)
    self.robustness = subsequences.map { $0.robustness * Double($0.Nsa) }.reduce(0, +)
      / Double(subsequences.map { $0.Nsa }.reduce(0, +))
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
public struct TrackerMetrics {
  /// Accuracy as defined in [1], equation (5).
  public let accuracy: Double

  /// Robustness as defined in [1], equation (6).
  public let robustness: Double

  public init(_ sequences: [SequenceMetrics]) {
    self.accuracy = sequences.map { $0.accuracy * Double($0.NFs) }.reduce(0, +)
      / Double(sequences.map { $0.NFs }.reduce(0, +))
    self.robustness = sequences.map { $0.robustness * Double($0.Ns) }.reduce(0, +)
      / Double(sequences.map { $0.Ns }.reduce(0, +))
  }
}

/// Expected Averge Overlap (EAO) for a tracker, as defined in [1].
public struct ExpectedAverageOverlap {
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
      curve.append(totalAverageOverlap / Double(availableSubsequenceCount))
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
  /// Prameter sequenceCount: How many sequences from `self` to use during the evaluation.
  public func evaluate(_ tracker: Tracker, sequenceCount: Int) -> TrackerEvaluationResults {
    let sequenceEvaluations = sequences.prefix(sequenceCount).enumerated().map {
      (i, sequence) -> SequenceEvaluationResults in
      print("Evaluating sequence \(i + 1) of \(sequences.count)")
      return sequence.evaluate(tracker)
    }
    return TrackerEvaluationResults(
      sequences: sequenceEvaluations,
      trackerMetrics: TrackerMetrics(sequenceEvaluations.map { $0.sequenceMetrics }),
      expectedAverageOverlap: ExpectedAverageOverlap(
        sequenceEvaluations.flatMap { $0.subsequences }.map { $0.metrics }))
  }
}

/// A single sequence in a `TrackerEvaluationDataset`.
public struct TrackerEvaluationSequence {
  /// The frames of the sequence.
  public let frames: [Tensor<Double>]

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
  public func evaluate(_ tracker: Tracker) -> SequenceEvaluationResults {
    let subsequences = self.subsequences(deltaAnchor: 50)
    let subsequenceEvaluations = subsequences.enumerated().map {
      (i, subsequence) -> (metrics: SubsequenceMetrics, prediction: [OrientedBoundingBox]) in
      print("Evaluating subsequence \(i + 1) of \(subsequences.count)")
      return subsequence.evaluateSubsequence(tracker)
    }
    return SequenceEvaluationResults(
      subsequences: subsequenceEvaluations,
      sequenceMetrics: SequenceMetrics(subsequenceEvaluations.map { $0.metrics }))
  }

  /// Returns the performance of `tracker` on the subsequence `self`.
  public func evaluateSubsequence(_ tracker: Tracker)
    -> (metrics: SubsequenceMetrics, prediction: [OrientedBoundingBox])
  {
    guard let _ = try? Python.attemptImport("shapely") else {
      print("python shapely library must be installed")
      preconditionFailure()
    }

    let prediction = tracker(frames, groundTruth[0])
    return (
      metrics: SubsequenceMetrics(groundTruth: groundTruth, prediction: prediction),
      prediction: prediction)
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
          video.frames[track.startFrameIndex..<(track.startFrameIndex + track.boxes.count)]),
        groundTruth: track.boxes)
      sequences.append(sequence)
    }
    self.sequences = sequences
  }
}

/// All the tracker evaluation metrics in one struct.
public struct TrackerEvaluationResults {
  /// The sequence results for all the sequences in the dataset.
  public let sequences: [SequenceEvaluationResults]

  /// The overall tracker metrics on this dataset.
  public let trackerMetrics: TrackerMetrics

  /// The overall expected average overlap curve for the tracker on this dataset.
  public let expectedAverageOverlap: ExpectedAverageOverlap
}

/// All the sequence evaluation metrics in one struct.
public struct SequenceEvaluationResults {
  /// The subsequence metrics for all subsequences in this sequence. And the predictions.
  public let subsequences: [(metrics: SubsequenceMetrics, prediction: [OrientedBoundingBox])]

  /// The sequence metrics for this sequence.
  public let sequenceMetrics: SequenceMetrics
}

/// Given `frames` and a `start` region containing an object to track, returns predicted regions
/// for all `frames` (including the first one).
public typealias Tracker =
  (_ frames: [Tensor<Double>], _ start: OrientedBoundingBox) -> [OrientedBoundingBox]
