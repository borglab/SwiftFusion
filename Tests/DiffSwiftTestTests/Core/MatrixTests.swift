import XCTest

import SwiftRT

class MatrixTests: XCTestCase {
    static var allTests = [
        ("test_concat", testConcat),
        ("test_log", test_log),
        ("test_neg", test_neg),
        ("test_squared", test_squared),
    ]

    //--------------------------------------------------------------------------
    // testConcat
    func testConcat() {
        let t1 = MatrixType<Float>(2, 3, with: 1...6)
        let t2 = MatrixType<Float>(2, 3, with: 7...12)
        let c1 = t1.concat(t2)
        XCTAssert(c1.extents == [4, 3])
        let c1Expected: [Float] = [
            1,  2,  3,
            4,  5,  6,
            7,  8,  9,
            10, 11, 12,
        ]
        XCTAssert(c1.flatArray == c1Expected)
        
        let c2 = t1.concat(t2, alongAxis: 1)
        XCTAssert(c2.extents == [2, 6])
        let c2Expected: [Float] = [
            1, 2, 3,  7,  8,  9,
            4, 5, 6, 10, 11, 12
        ]
        XCTAssert(c2.flatArray == c2Expected)
    }

    //--------------------------------------------------------------------------
    // test_log
    func test_log() {
        let range = 0..<6
        let matrix = MatrixType<Float>(3, 2, with: range)
        let values = log(matrix).flatArray
        let expected: [Float] = range.map { Foundation.log(Float($0)) }
        XCTAssert(values == expected)
    }
    
    //--------------------------------------------------------------------------
    // test_neg
    func test_neg() {
        let range = 0..<6
        let matrix = MatrixType<Float>(3, 2, with: range)
        let expected: [Float] = range.map { -Float($0) }

        let values = matrix.neg().flatArray
        XCTAssert(values == expected)
        
        let values2 = -matrix
        XCTAssert(values2.flatArray == expected)
    }
    
    //--------------------------------------------------------------------------
    // test_squared
    func test_squared() {
        let matrix = MatrixType<Float>(3, 2, with: [0, -1, 2, -3, 4, 5])
        let values = matrix.squared().flatArray
        let expected: [Float] = (0...5).map { Float($0 * $0) }
        XCTAssert(values == expected)
    }
        
    //--------------------------------------------------------------------------
    // test_multiplication
    func test_multiplication() {
        let matrix = MatrixType<Float>(3, 2, with: [0, -1, 2, -3, 4, 5])
        let values = (matrix * matrix).flatArray
        let expected: [Float] = (0...5).map { Float($0 * $0) }
        XCTAssert(values == expected)
    }
}