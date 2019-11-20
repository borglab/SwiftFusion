import XCTest

#if !canImport(ObjectiveC)
public func allTests() -> [XCTestCaseEntry] {
  [
    testCase(DiffSwiftTestTests.allTests),
    testCase(Pose2Tests.allTests),
  ]
}
#endif
