/// Pose2 class is the new Swift type for the SE(2) manifold of 2D Euclidean
/// Poses
public struct Pose2: Equatable, Differentiable, KeyPathIterable, TangentStandardBasis {
  private var t_: Point2
  private var rot_: Rot2

  @differentiable public var t: Point2 { t_ }
  @differentiable public var rot: Rot2 { rot_ }

  public struct TangentVector:
    Differentiable, AdditiveArithmetic, KeyPathIterable, VectorProtocol, ElementaryFunctions
  {
    public typealias VectorSpaceScalar = Double
    public var t: Point2.TangentVector
    public var rot: Rot2.TangentVector
    public init(t: Point2.TangentVector, rot: Rot2.TangentVector) {
      self.t = t
      self.rot = rot
    }
  }
  public mutating func move(along direction: TangentVector) {
    // TODO: This should be the real exponential map.
    self.t_.move(along: self.rot_.rotate(direction.t))
    self.rot_.move(along: direction.rot)
  }

  @differentiable
  public init(_ x: Double, _ y: Double, _ theta: Double) {
    self.init(Rot2(theta), Point2(x, y))
  }

  @differentiable
  public init(_ r: Rot2, _ t: Point2) {
    t_ = t
    rot_ = r
  }

  @derivative(of: t)
  @usableFromInline
  func vjpT() -> (value: Point2, pullback: (Point2.TangentVector) -> TangentVector) {
    return (t_, { TangentVector(t: rot_.unrotate($0), rot: .zero) })
  }

  @derivative(of: rot)
  @usableFromInline
  func vjpRot() -> (value: Rot2, pullback: (Rot2.TangentVector) -> TangentVector) {
    return (rot_, { TangentVector(t: .zero, rot: $0) })
  }

  @derivative(of: init(_:_:))
  @usableFromInline
  static func vjpInit(_ r: Rot2, _ t: Point2)
    -> (value: Pose2, pullback: (TangentVector) -> (Rot2.TangentVector, Point2.TangentVector))
  {
    return (Pose2(r, t), { ($0.rot, r.rotate($0.t)) })
  }
}

extension Pose2 {
  /// The adjoint representation of `self`.
  public func adjoint(_ v: TangentVector) -> TangentVector {
    TangentVector(
      t: rot.rotate(v.t) - Point2.TangentVector(x: -t.y * v.rot, y: t.x * v.rot),
      rot: v.rot
    )
  }
}

extension Pose2 {
  public static func == (lhs: Pose2, rhs: Pose2) -> Bool {
    (lhs.t, lhs.rot) == (rhs.t, rhs.rot)
  }

  @differentiable
  public static func * (a: Pose2, b: Pose2) -> Pose2 {
    Pose2(a.rot * b.rot, a.t + a.rot * b.t)
  }
}

@differentiable
public func inverse(_ p: Pose2) -> Pose2 {
  Pose2(inverse(p.rot), p.rot.unrotate(-p.t))
}

/// Calculate relative pose 1T2 between two poses wT1 and wT2
@differentiable
public func between(_ wT1: Pose2, _ wT2: Pose2) -> Pose2 {
  inverse(wT1) * wT2
}
