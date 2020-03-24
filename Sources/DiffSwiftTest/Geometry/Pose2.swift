/// Pose2 class is the new Swift type for the SE(2) manifold of 2D Euclidean
/// Poses
public struct Pose2: Equatable, Differentiable, KeyPathIterable {
  public var t_: Point2
  public var rot_: Rot2

  @differentiable
  public init(_ x: Double, _ y: Double, _ theta: Double) {
    t_ = Point2(x, y)
    rot_ = Rot2(theta)
  }

  @differentiable
  public init(_ r: Rot2, _ t: Point2) {
    t_ = t
    rot_ = r
  }

  public static func == (lhs: Pose2, rhs: Pose2) -> Bool {
    (lhs.t_, lhs.rot_) == (rhs.t_, rhs.rot_)
  }

  public static func * (a: Pose2, b: Pose2) -> Pose2 {
    Pose2(a.rot_ * b.rot_, a.t_ + a.rot_ * b.t_)
  }
}

@differentiable
func inverse(_ p: Pose2) -> Pose2 {
  Pose2(inverse(p.rot_), p.rot_.unrotate(-p.t_))
}

/// Calculate relative pose 1T2 between two poses wT1 and wT2
@differentiable
public func between(_ wT1: Pose2, _ wT2: Pose2) -> Pose2 {
  inverse(wT1) * wT2
}