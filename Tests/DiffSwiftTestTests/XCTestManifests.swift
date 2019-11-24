import XCTest

#if !canImport(ObjectiveC)
public func allTests() -> [XCTestCaseEntry] {
  [
    testCase(Rot2Tests.allTests),
    testCase(Pose2Tests.allTests),
  ]
}
#endif
