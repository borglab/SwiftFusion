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
  let _ = x.recursivelyAllKeyPaths(to: Double.self).map {
    XCTAssert(
      abs(x[keyPath: $0] - y[keyPath: $0]) < accuracy,
      "value mismatch:\n\(x)\nis not equal to\n\(y)\nwith accuracy \(accuracy)",
      file: file,
      line: line
    )
  }
}

/// Create a `Tensor<Double>` with shape (2, 1)
/// TODO: replace with the `Vector2` Marc prototyped
public func Vector2_t(_ x: Double, _ y: Double) -> Tensor<Double> {
  let t = Tensor<Double>(shape: [2, 1], scalars: [x, y])
  return t
}

/// Factor graph with 2 2D factors on 3 2D variables
public final class SimpleGaussianFactorGraph {
  public static func create() -> GaussianFactorGraph {
    var fg = GaussianFactorGraph()
    
    let I_2x2: Tensor<Double> = eye(rowCount: 2)
    let x1 = 2, x2 = 0, l1 = 1
    
    // linearized prior on x1: c[_x1_]+x1=0 i.e. x1=-c[_x1_]
    fg += JacobianFactor([x1], [10 * I_2x2], -1.0 * Vector2_t(1.0, 1.0))
    // odometry between x1 and x2: x2-x1=[0.2;-0.1]
    fg += JacobianFactor([x2, x1], [10 * I_2x2, -10 * I_2x2], Vector2_t(2.0, -1.0))
    // measurement between x1 and l1: l1-x1=[0.0;0.2]
    fg += JacobianFactor([l1, x1], [5 * I_2x2, -5 * I_2x2], Vector2_t(0.0, 1.0))
    // measurement between x2 and l1: l1-x2=[-0.2;0.3]
    fg += JacobianFactor([x2, l1], [-5 * I_2x2, 5 * I_2x2], Vector2_t(-1.0, 1.5))
    return fg;
  }

  public static func correctDelta() -> VectorValues {
    let x1 = 2, x2 = 0, l1 = 1
    var c = VectorValues()
    c.insert(l1, Vector2_t(-0.1, 0.1))
    c.insert(x1, Vector2_t(-0.1, -0.1))
    c.insert(x2, Vector2_t(0.1, -0.2))
    
    return c
  }

  public static func zeroDelta() -> VectorValues {
    let x1 = 2, x2 = 0, l1 = 1
    var c = VectorValues()
    c.insert(l1, Vector2_t(0.0, 0.0))
    c.insert(x1, Vector2_t(0.0, 0.0))
    c.insert(x2, Vector2_t(0.0, 0.0))
    
    return c
  }
}

extension URL {
  /// Creates a URL for the directory containing the caller's source file.
  static func sourceFileDirectory(file: String = #file) -> URL {
    return URL(fileURLWithPath: file).deletingLastPathComponent()
  }
}
