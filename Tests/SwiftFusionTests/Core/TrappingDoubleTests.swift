// Copyright 2020 The SwiftFusion Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import XCTest

import PenguinStructures
import SwiftFusion

class TrappingDoubleTests: XCTestCase {
  /// Tests simple `TrappingDouble` operations.
  func testOperations() {
    let x: TrappingDouble = 1.0
    let y: TrappingDouble = 2.0
    XCTAssertEqual(y + x, 3.0)
    XCTAssertEqual(y - x, 1.0)
  }

  /// Tests that `TrappingDouble` traps on `NaN`s.
  ///
  /// There is no way to assert that a trap happens, so this test is skipped. You can remove the
  /// `XCTSkipIf` line and run this test to manually verify the trapping behavior.
  func testTrap() throws {
    try XCTSkipIf(true)
    let _: TrappingDouble = .infinity - .infinity
  }
}
