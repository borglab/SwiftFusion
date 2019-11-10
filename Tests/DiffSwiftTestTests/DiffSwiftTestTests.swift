import XCTest
@testable import DiffSwiftTest

final class DiffSwiftTestTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(DiffSwiftTest().text, "Hello, World!")
    }
    
    func testTempRot2() {
        let testRot2 = TestRot2()
        testRot2.testBetweenIdentities();
        testRot2.testBetweenIdentitiesTrivial();
        testRot2.testBetweenDerivatives();
    }
    
    static var allTests = [
        ("testExample", testExample),
        ("testTempRot2", testTempRot2)
    ]
}
