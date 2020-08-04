// WARNING: This is a generated file. Do not edit it. Instead, edit the corresponding ".gyb" file.
// See "generate.sh" in the root of this repository for instructions how to regenerate files.

// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 1)
import Foundation
import TensorFlow
import XCTest

import SwiftFusion

// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 9)

class ConcreteVectorTests: XCTestCase {
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 19)

  /// Test that initializing a vector from coordinate values works.
  func testVector1Init() {
    let vector1 = Vector1(1)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.x, 1)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 26)
  }

// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 19)

  /// Test that initializing a vector from coordinate values works.
  func testVector2Init() {
    let vector1 = Vector2(1, 2)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.x, 1)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.y, 2)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 26)
  }

// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 19)

  /// Test that initializing a vector from coordinate values works.
  func testVector3Init() {
    let vector1 = Vector3(1, 2, 3)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.x, 1)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.y, 2)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.z, 3)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 26)
  }

// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 19)

  /// Test that initializing a vector from coordinate values works.
  func testVector4Init() {
    let vector1 = Vector4(1, 2, 3, 4)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s0, 1)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s1, 2)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s2, 3)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s3, 4)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 26)
  }

// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 19)

  /// Test that initializing a vector from coordinate values works.
  func testVector5Init() {
    let vector1 = Vector5(1, 2, 3, 4, 5)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s0, 1)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s1, 2)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s2, 3)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s3, 4)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s4, 5)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 26)
  }

// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 19)

  /// Test that initializing a vector from coordinate values works.
  func testVector6Init() {
    let vector1 = Vector6(1, 2, 3, 4, 5, 6)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s0, 1)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s1, 2)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s2, 3)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s3, 4)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s4, 5)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s5, 6)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 26)
  }

// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 19)

  /// Test that initializing a vector from coordinate values works.
  func testVector7Init() {
    let vector1 = Vector7(1, 2, 3, 4, 5, 6, 7)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s0, 1)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s1, 2)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s2, 3)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s3, 4)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s4, 5)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s5, 6)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s6, 7)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 26)
  }

// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 19)

  /// Test that initializing a vector from coordinate values works.
  func testVector8Init() {
    let vector1 = Vector8(1, 2, 3, 4, 5, 6, 7, 8)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s0, 1)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s1, 2)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s2, 3)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s3, 4)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s4, 5)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s5, 6)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s6, 7)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s7, 8)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 26)
  }

// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 19)

  /// Test that initializing a vector from coordinate values works.
  func testVector9Init() {
    let vector1 = Vector9(1, 2, 3, 4, 5, 6, 7, 8, 9)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s0, 1)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s1, 2)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s2, 3)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s3, 4)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s4, 5)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s5, 6)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s6, 7)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s7, 8)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 24)
    XCTAssertEqual(vector1.s8, 9)
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 26)
  }

// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 29)
}

// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 32)
/// Tests the `Vector` requirements.
class Vector1VectorTests: XCTestCase, FixedSizeVectorTests {
  typealias Testee = Vector1
  static var dimension: Int { return 1 }
  func testAll() {
    runAllFixedSizeVectorTests()
  }
}
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 32)
/// Tests the `Vector` requirements.
class Vector2VectorTests: XCTestCase, FixedSizeVectorTests {
  typealias Testee = Vector2
  static var dimension: Int { return 2 }
  func testAll() {
    runAllFixedSizeVectorTests()
  }
}
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 32)
/// Tests the `Vector` requirements.
class Vector3VectorTests: XCTestCase, FixedSizeVectorTests {
  typealias Testee = Vector3
  static var dimension: Int { return 3 }
  func testAll() {
    runAllFixedSizeVectorTests()
  }
}
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 32)
/// Tests the `Vector` requirements.
class Vector4VectorTests: XCTestCase, FixedSizeVectorTests {
  typealias Testee = Vector4
  static var dimension: Int { return 4 }
  func testAll() {
    runAllFixedSizeVectorTests()
  }
}
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 32)
/// Tests the `Vector` requirements.
class Vector5VectorTests: XCTestCase, FixedSizeVectorTests {
  typealias Testee = Vector5
  static var dimension: Int { return 5 }
  func testAll() {
    runAllFixedSizeVectorTests()
  }
}
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 32)
/// Tests the `Vector` requirements.
class Vector6VectorTests: XCTestCase, FixedSizeVectorTests {
  typealias Testee = Vector6
  static var dimension: Int { return 6 }
  func testAll() {
    runAllFixedSizeVectorTests()
  }
}
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 32)
/// Tests the `Vector` requirements.
class Vector7VectorTests: XCTestCase, FixedSizeVectorTests {
  typealias Testee = Vector7
  static var dimension: Int { return 7 }
  func testAll() {
    runAllFixedSizeVectorTests()
  }
}
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 32)
/// Tests the `Vector` requirements.
class Vector8VectorTests: XCTestCase, FixedSizeVectorTests {
  typealias Testee = Vector8
  static var dimension: Int { return 8 }
  func testAll() {
    runAllFixedSizeVectorTests()
  }
}
// ###sourceLocation(file: "Tests/SwiftFusionTests/Core/VectorNTests.swift.gyb", line: 32)
/// Tests the `Vector` requirements.
class Vector9VectorTests: XCTestCase, FixedSizeVectorTests {
  typealias Testee = Vector9
  static var dimension: Int { return 9 }
  func testAll() {
    runAllFixedSizeVectorTests()
  }
}
