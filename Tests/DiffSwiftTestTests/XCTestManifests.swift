import XCTest

#if !canImport(ObjectiveC)
public func allTests() -> [XCTestCaseEntry] {
  [
    testCase(DiffSwiftTestTests.allTests),
  ]
}
#endif
