import _Differentiation
import XCTest

import TensorFlow
import SwiftFusion

class TensorFlowMatrixTests: XCTestCase {
    //--------------------------------------------------------------------------
    // testConcat
    func testConcat() {
        let t1 = Tensor<Double>(shape: [2, 3], scalars: (1...6).map { Double($0) })
        let t2 = Tensor<Double>(shape: [2, 3], scalars: (7...12).map { Double($0) })
        let c1 = t1.concatenated(with: t2)
        let c1Expected = Tensor<Double>([
            1,  2,  3,
            4,  5,  6,
            7,  8,  9,
            10, 11, 12,
        ])
        
        XCTAssert(c1.flattened() == c1Expected)
        
        let c2 = t1.concatenated(with: t2, alongAxis: 1)
        let c2Expected = Tensor<Double>([
            1, 2, 3,  7,  8,  9,
            4, 5, 6, 10, 11, 12
        ])
        
        XCTAssert(c2.flattened() == c2Expected)
    }

    //--------------------------------------------------------------------------
    // test_log
    func test_log() {
        let range = 0..<6
        let matrix = Tensor(shape: [3, 2], scalars: (range).map { Double($0) })
        let values = log(matrix).flattened()
        let expected = Tensor(range.map { log(Double($0)) })
        assertEqual(values, expected, accuracy: 1e-8)
    }
    
    //--------------------------------------------------------------------------
    // test_neg
    func test_neg() {
        let range = 0..<6
        let matrix = Tensor(shape: [3, 2], scalars: (range).map { Double($0) })
        let expected = Tensor(range.map { -Double($0) })

        let values = (-matrix).flattened()
        assertEqual(values, expected, accuracy: 1e-8)
    }
    
    //--------------------------------------------------------------------------
    // test_squared
    func test_squared() {
        let matrix = Tensor(shape: [3, 2], scalars: ([0, -1, 2, -3, 4, 5]).map { Double($0) })
        let values = matrix.squared().flattened()
        let expected = Tensor((0...5).map { Double ($0 * $0) })
        assertEqual(values, expected, accuracy: 1e-8)
    }
        
    //--------------------------------------------------------------------------
    // test_multiplication
    func test_multiplication() {
        let matrix = Tensor(shape: [3, 2], scalars: ([0, -1, 2, -3, 4, 5]).map { Double($0) })
        let values = (matrix * matrix).flattened()
        let expected = Tensor((0...5).map { Double($0 * $0) })
        assertEqual(values, expected, accuracy: 1e-8)
    }
}
