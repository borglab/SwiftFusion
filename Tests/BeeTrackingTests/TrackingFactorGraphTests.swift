import XCTest
import BeeTracking
import PythonKit
import PenguinStructures
import BeeDataset
import BeeTracking
import SwiftFusion
// import TensorFlow

class TrackingFactorGraphTests: XCTestCase {
  let datasetDirectory = URL.sourceFileDirectory().appendingPathComponent("../BeeDatasetTests/fakeDataset")

  func testGetTrainingBatches() {
    let dataset = OISTBeeVideo(directory: datasetDirectory, length: 2)!

    let (fg, bg, _) = getTrainingBatches(
      dataset: dataset, boundingBoxSize: (40, 70),
      fgBatchSize: 10, bgBatchSize: 11, 
      fgRandomFrameCount: 1,
      bgRandomFrameCount: 1
    )

    XCTAssertEqual(fg.shape, TensorShape([10, 40, 70, 1]))
    XCTAssertEqual(bg.shape, TensorShape([11, 40, 70, 1]))
  }

  func testTrainRPTracker() {
    let trainingData = OISTBeeVideo(directory: datasetDirectory, length: 1)!
    let testData = OISTBeeVideo(directory: datasetDirectory, length: 2)!

    // Test regular training
    var tracker : TrackingConfiguration<Tuple1<Pose2>> = trainRPTracker(
      trainingData: trainingData,
      frames: testData.frames, boundingBoxSize: (40, 70), withFeatureSize: 100,
      fgRandomFrameCount: 1,
      bgRandomFrameCount: 1
    )
    XCTAssertEqual(tracker.frameVariableIDs.count, 2)

    // Test training with Monte Carlo EM
    tracker = trainRPTracker(
      trainingData: trainingData,
      frames: testData.frames, boundingBoxSize: (40, 70), withFeatureSize: 100,
      fgRandomFrameCount: 1, bgRandomFrameCount: 1, usingEM: true
    )
    XCTAssertEqual(tracker.frameVariableIDs.count, 2)
  }

  func testCreateSingleRPTrack() {
    let trainingData = OISTBeeVideo(directory: datasetDirectory, length: 2)!
    let testData = OISTBeeVideo(directory: datasetDirectory, length: 2)!
    var tracker = trainRPTracker(
      trainingData: trainingData,
      frames: testData.frames, boundingBoxSize: (40, 70), withFeatureSize: 100,
      fgRandomFrameCount: 2,
      bgRandomFrameCount: 2
    )
    
    var (track, groundTruth) = createSingleTrack(
      onTrack: 0, withTracker: &tracker, andTestData: testData
    )
    XCTAssertEqual(track.count, 2)
    XCTAssertEqual(groundTruth.count, 2)
    
    // Now try with sampling
    (track, groundTruth) = createSingleTrack(
      onTrack: 0, withTracker: &tracker, andTestData: testData, withSampling: true
    )
    XCTAssertEqual(track.count, 2)
    XCTAssertEqual(groundTruth.count, 2)
  }

  // func testRunRPTracker() {
  //   let fig: PythonObject = runRPTracker(onTrack: 15)
  //   XCTAssertEqual(fig.axes.count, 2)
  // }
}

extension URL {
  /// Creates a URL for the directory containing the caller's source file.
  fileprivate static func sourceFileDirectory(file: String = #filePath) -> URL {
    return URL(fileURLWithPath: file).deletingLastPathComponent()
  }
}
