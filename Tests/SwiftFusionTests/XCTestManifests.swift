import XCTest

#if !canImport(ObjectiveC)
public func allTests() -> [XCTestCaseEntry] {
  [
    testCase(AnyDifferentiableTests.allTests),
    testCase(DictionaryDifferentiableTests.allTests),
    testCase(G2OReaderTests.allTests),
    testCase(Rot2Tests.allTests),
    testCase(Pose2Tests.allTests),
    testCase(JacobianTests.allTests),
    testCase(VectorTests.allTests),
  ]
}
#endif
