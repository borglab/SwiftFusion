import Foundation
import XCTest

@testable import DiffSwiftTest

class JacobianProtocolTests: XCTestCase {
  static var allTests = [
    ("testJacobianPose2Identity", testJacobianPose2Identity),
    ("testJacobianPose2Trivial", testJacobianPose2Trivial),
  ]

  // --------------------------------------------------------------------------
  // testConcat
  func testJacobianPose2Identity() {
    let wT1 = Pose2(1, 0, 3.1415926 / 2.0), wT2 = Pose2(1, 0, 3.1415926 / 2.0)
    let map: [Pose2] = [wT1, wT2]

    let ef: @differentiable(_ map: [Pose2]) -> Double = { (_ map: [Pose2]) -> Double in
      let d = between(map[0], map[1])

      return d.rot_.theta * d.rot_.theta + d.t_.x * d.t_.x + d.t_.y * d.t_.y
    }

    let j = jacobian(of: ef, at: map, basisVectors: Array(repeating: 1.0, count: 1))
    print("J(ef) = \(j[0].base as AnyObject)")
    for item in j[0] {
      XCTAssertEqual(item, Pose2.TangentVector.zero)
    }
  }

  func testJacobianPose2Trivial() {
    let wT1 = Pose2(1, 0, 3.1415926 / 2.0), wT2 = Pose2(1, 1, -3.1415926 / 2.0)
    let map: [Pose2] = [wT1, wT2]

    let ef: @differentiable(_ map: [Pose2]) -> Double = { (_ map: [Pose2]) -> Double in
      let d = between(map[0], map[1])

      return d.rot_.theta * d.rot_.theta + d.t_.x * d.t_.x + d.t_.y * d.t_.y
    }

    let j = jacobian(of: ef, at: map, basisVectors: Array(repeating: 1.0, count: 1))
    print("J(ef) = \(j[0].base as AnyObject)")

    print("Pose2.TangentVector.basisVectors() = \(Pose2.TangentVector.basisVectors() as AnyObject)")
  }
}
