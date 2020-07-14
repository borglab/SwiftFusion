import PenguinStructures
import TensorFlow
import SwiftFusion
import XCTest

/// Asserts that `x` and `y` have the same shape and that their values have absolute difference
/// less than `accuracy`.
func assertEqual<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>, _ y: Tensor<T>, accuracy: T, file: StaticString = #file, line: UInt = #line
) {
  guard x.shape == y.shape else {
    XCTFail(
      "shape mismatch: \(x.shape) is not equal to \(y.shape)",
      file: file,
      line: line
    )
    return
  }
  XCTAssert(
    abs(x - y).max().scalarized() < accuracy,
    "value mismatch:\n\(x)\nis not equal to\n\(y)\nwith accuracy \(accuracy)",
    file: file,
    line: line
  )
}

/// Asserts that `x` and `y` have absolute difference less than `accuracy`.
func assertAllKeyPathEqual<T: KeyPathIterable>(
  _ x: T, _ y: T, accuracy: Double, file: StaticString = #file, line: UInt = #line
) {
  let result: [Bool] = x.recursivelyAllKeyPaths(to: Double.self).map {
    if !(abs(x[keyPath: $0] - y[keyPath: $0]) < accuracy) {
      return false
    } else {
      return true
    }
  }
  if !result.allSatisfy({ x in x == true }) {
    XCTAssert(
      false,
      "value mismatch:\n\(x)\nis not equal to\n\(y)\nwith accuracy \(accuracy)",
      file: file,
      line: line
    )
  }
}

/// Factor graph with 2 2D factors on 3 2D variables
public final class SimpleOldGaussianFactorGraph {
  public static func create() -> OldGaussianFactorGraph {
    var fg = OldGaussianFactorGraph()
    
    let I_2x2 = Matrix(eye: 2)
    let x1 = 2, x2 = 0, l1 = 1
    
    // linearized prior on x1: c[_x1_]+x1=0 i.e. x1=-c[_x1_]
    fg += OldJacobianFactor([x1], [10 * I_2x2], -1.0 * Vector([1.0, 1.0]))
    // odometry between x1 and x2: x2-x1=[0.2;-0.1]
    fg += OldJacobianFactor([x2, x1], [10 * I_2x2, -10 * I_2x2], Vector([2.0, -1.0]))
    // measurement between x1 and l1: l1-x1=[0.0;0.2]
    fg += OldJacobianFactor([l1, x1], [5 * I_2x2, -5 * I_2x2], Vector([0.0, 1.0]))
    // measurement between x2 and l1: l1-x2=[-0.2;0.3]
    fg += OldJacobianFactor([x2, l1], [-5 * I_2x2, 5 * I_2x2], Vector([-1.0, 1.5]))
    return fg;
  }

  public static func correctDelta() -> VectorValues {
    let x1 = 2, x2 = 0, l1 = 1
    var c = VectorValues()
    c.insert(l1, Vector([-0.1, 0.1]))
    c.insert(x1, Vector([-0.1, -0.1]))
    c.insert(x2, Vector([0.1, -0.2]))
    
    return c
  }

  public static func zeroDelta() -> VectorValues {
    let x1 = 2, x2 = 0, l1 = 1
    var c = VectorValues()
    c.insert(l1, Vector([0.0, 0.0]))
    c.insert(x1, Vector([0.0, 0.0]))
    c.insert(x2, Vector([0.0, 0.0]))
    
    return c
  }
}

/// Factor graph with 2 2D factors on 3 2D variables.
public enum SimpleGaussianFactorGraph {
  public static let x1ID = TypedID<Vector2>(2)
  public static let x2ID = TypedID<Vector2>(0)
  public static let l1ID = TypedID<Vector2>(1)

  public static func create() -> GaussianFactorGraph {
    let I_2x2 = FixedSizeMatrix2.identity
    var fg = GaussianFactorGraph(zeroValues: zeroDelta())
    fg.store(JacobianFactor2x2_1(
      jacobian: 10 * I_2x2,
      error: -1 * Vector2(1, 1),
      edges: Tuple1(x1ID)))
    fg.store(JacobianFactor2x2_2(
      jacobians: 10 * I_2x2, -10 * I_2x2,
      error: Vector2(2, -1),
      edges: Tuple2(x2ID, x1ID)))
    fg.store(JacobianFactor2x2_2(
      jacobians: 5 * I_2x2, -5 * I_2x2,
      error: Vector2(0, 1),
      edges: Tuple2(l1ID, x1ID)))
    fg.store(JacobianFactor2x2_2(
      jacobians: -5 * I_2x2, 5 * I_2x2,
      error: Vector2(-1, 1.5),
      edges: Tuple2(x2ID, l1ID)))
    return fg
  }

  public static func correctDelta() -> VariableAssignments {
    var x = VariableAssignments()
    let actualX2ID = x.store(Vector2(0.1, -0.2))
    assert(actualX2ID == x2ID)
    let actualL1ID = x.store(Vector2(-0.1, 0.1))
    assert(actualL1ID == l1ID)
    let actualX1ID = x.store(Vector2(-0.1, -0.1))
    assert(actualX1ID == x1ID)
    return x
  }

  public static func zeroDelta() -> VariableAssignments {
    var x = VariableAssignments()
    let actualX2ID = x.store(Vector2.zero)
    assert(actualX2ID == x2ID)
    let actualL1ID = x.store(Vector2.zero)
    assert(actualL1ID == l1ID)
    let actualX1ID = x.store(Vector2.zero)
    assert(actualX1ID == x1ID)
    return x
  }
}

extension URL {
  /// Creates a URL for the directory containing the caller's source file.
  static func sourceFileDirectory(file: String = #file) -> URL {
    return URL(fileURLWithPath: file).deletingLastPathComponent()
  }
}
