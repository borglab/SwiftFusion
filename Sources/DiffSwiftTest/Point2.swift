import TensorFlow

/// Point2 class is the new Swift type for the 2D vector space R^2
struct Point2: Equatable, Differentiable {
  var x_, y_: Double

  @differentiable
  init(_ x: Double, _ y: Double) {
    (x_, y_) = (x, y)
  }

  // Differentiable computed property.
  @differentiable // Implicitly: @differentiable(wrt: self)
  var magnitude: Double {
    (x_ * x_ + y_ * y_).squareRoot()
  }

  public static func == (lhs: Point2, rhs: Point2) -> Bool {
    (lhs.x_, lhs.y_) == (rhs.x_, rhs.y_)
  }

  @differentiable
  public static func + (a: Point2, b: Point2) -> Point2 {
    Point2(a.x_ + b.x_, a.y_ + b.y_)
  }

  public static prefix func - (a: Point2) -> Point2 {
    Point2(-a.x_, -a.y_)
  }
}
