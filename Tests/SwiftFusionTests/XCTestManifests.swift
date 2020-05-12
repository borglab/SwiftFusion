import XCTest

#if !canImport(ObjectiveC)
public func allTests() -> [XCTestCaseEntry] {
  [
    testCase(AnyDifferentiableTests.allTests),
    testCase(CGLSTests.allTests),
    testCase(DictionaryDifferentiableTests.allTests),
    testCase(G2OReaderTests.allTests),
    testCase(GaussianFactorGraphTests.allTests),
    testCase(NonlinearFactorGraphTests.allTests),
    testCase(MatrixTests.allTests),
    testCase(Pose2Tests.allTests),
    testCase(Rot2Tests.allTests),
    testCase(TensorFlowMatrixTests.allTests),
    testCase(VectorNTests.allTests),
    testCase(VectorTests.allTests),
  ]
}
#endif
