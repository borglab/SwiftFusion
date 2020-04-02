import XCTest

#if !canImport(ObjectiveC)
public func allTests() -> [XCTestCaseEntry] {
  [
    testCase(Rot2Tests.allTests),
    testCase(Pose2Tests.allTests),
    testCase(JacobianTests.allTests),
    testCase(NonlinearFactorGraphTests.allTests),
    testCase(VectorTests.allTests),
  ]
}
#endif
