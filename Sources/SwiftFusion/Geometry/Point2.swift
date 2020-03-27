import TensorFlow

/// Point2 class is the new Swift type for the 2D vector space R^2
public struct Point2: Equatable, Differentiable, JacobianEvaluatable {
  public var x, y: Double

  @differentiable
  public init(_ x: Double, _ y: Double) {
    (self.x, self.y) = (x, y)
  }

  // Differentiable computed property.
  @differentiable
  public var magnitude: Double {
    (x * x + y * y).squareRoot()
  }
  
  public static func == (lhs: Point2, rhs: Point2) -> Bool {
    (lhs.x, lhs.y) == (rhs.x, rhs.y)
  }

  @differentiable
  public static func + (a: Point2, b: Point2) -> Point2 {
    Point2(a.x + b.x, a.y + b.y)
  }

  @differentiable
  public static prefix func - (a: Point2) -> Point2 {
    Point2(-a.x, -a.y)
  }
  
  @differentiable
  public static func - (a: Point2, b: Point2) -> Point2 {
    Point2(a.x - b.x, a.y - b.y)
  }
}
