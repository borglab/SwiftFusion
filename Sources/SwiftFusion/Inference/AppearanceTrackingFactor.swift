import PenguinStructures
import TensorFlow

/// A factor over a target's pose and appearance in an image.
public struct AppearanceTrackingFactor<LatentCode: FixedSizeVector>: LinearizableFactor2 {
  /// The first adjacent variable, the pose of the target in the image.
  ///
  /// This explicitly specifies `LinearizableFactor2`'s `associatedtype V0`.
  public typealias V0 = Pose2

  /// The second adjacent variable, the latent code for the appearance of the target.
  ///
  /// This explicitly specifies `LinearizableFactor2`'s `associatedtype V1`.
  public typealias V1 = LatentCode

  /// A region cropped from the image.
  public typealias Patch = TensorVector

  /// The IDs of the variables adjacent to this factor.
  public let edges: Variables.Indices

  /// The image containing the target.
  public let measurement: Tensor<Double>

  /// A generative model that produces an appearance from a latent code.
  ///
  /// - Parameters:
  ///   - latentCode: the input to the generative model, which is a tensor with shape
  ///     `[LatentCode.dimension]`.
  /// - Returns:
  ///   - appearance: the output of the generative model, which is a tensor with shape
  ///     `Patch.shape`.
  ///   - jacobian: the Jacobian of the generative model at the given `latentCode`, which is a
  ///     tensor with shape `Patch.shape + [LatentCode.dimension]`.
  public typealias GenerativeModel =
    (_ latentCode: Tensor<Double>) -> (appearance: Tensor<Double>, jacobian: Tensor<Double>)

  /// The generative model that produces an appearance from a latent code.
  public var appearanceModel: GenerativeModel

  /// Creates an instance.
  ///
  /// - Parameters:
  ///   - poseId: the id of the adjacent pose variable.
  ///   - latentId: the id of the adjacent latent code variable.
  ///   - measurement: the image containing the target.
  ///   - appearanceModel: the generative model that produces an appearance from a latent code.
  public init(
    _ poseId: TypedID<Pose2>,
    _ latentId: TypedID<LatentCode>,
    measurement: Tensor<Double>,
    appearanceModel: @escaping GenerativeModel
  ) {
    self.edges = Tuple2(poseId, latentId)
    self.measurement = measurement
    self.appearanceModel = appearanceModel
  }

  /// Returns the difference between the PPCA generated `Patch` and the `Patch` cropped from
  /// `measurement`.
  @differentiable
  public func errorVector(_ pose: Pose2, _ latent: LatentCode) -> Patch.TangentVector {
    let (appearance, _) = appearanceModel(latent.flatTensor)
    let region = OrientedBoundingBox(
      center: pose, rows: appearance.shape[0], cols: appearance.shape[1])
    return Patch(appearance - measurement.patch(at: region))
  }

  @derivative(of: errorVector)
  @usableFromInline
  func vjpErrorVector(_ pose: Pose2, _ latent: LatentCode) -> (value: Patch.TangentVector, pullback: (Patch.TangentVector) -> (Pose2.TangentVector, LatentCode)) {
    fatalError("not implemented")
  }

  /// Returns a linear approximation to `self` at `x`.
  public func linearized(at x: Variables) -> LinearizedAppearanceTrackingFactor<LatentCode> {
    let pose = x.head
    let latent = x.tail.head
    let (generatedAppearance, generatedAppearance_H_latent) = appearanceModel(latent.flatTensor)
    let region = OrientedBoundingBox(
      center: pose, rows: generatedAppearance.shape[0], cols: generatedAppearance.shape[1])
    let (actualAppearance, actualAppearance_H_pose) =
      measurement.patchWithJacobian(at: region)
    assert(generatedAppearance_H_latent.shape == generatedAppearance.shape + [LatentCode.dimension])
    return LinearizedAppearanceTrackingFactor<LatentCode>(
      error: Patch(actualAppearance - generatedAppearance),
      errorVector_H_pose: -actualAppearance_H_pose,
      errorVector_H_latent: generatedAppearance_H_latent,
      edges: Variables.linearized(edges))
  }

  /// Returns the linearizations of `factors` at `x`.
  ///
  /// Note: This causes factor graph linearization to use our custom linearization,
  /// `LinearizedPPCATrackingFactor` instead of the default AD-generated linearization.
  public static func linearized<C: Collection>(_ factors: C, at x: VariableAssignments)
    -> AnyGaussianFactorArrayBuffer where C.Element == Self
  {
     .init(Variables.withBufferBaseAddresses(x) { varsBufs in
       .init(factors.lazy.map { f in f.linearized(at: Variables(at: f.edges, in: varsBufs)) })
     })
  }
}

/// A linear approximation to `AppearanceTrackingFactor`, at a certain linearization point.
public struct LinearizedAppearanceTrackingFactor<LatentCode: FixedSizeVector>: GaussianFactor {

  /// The tangent vectors of the `AppearanceTrackingFactor`'s "pose" and "latent" variables.
  public typealias Variables = Tuple2<Pose2.TangentVector, LatentCode>

  /// A region cropped from the image.
  public typealias Patch = TensorVector

  /// The error vector at the linearization point.
  public let error: Patch

  /// The Jacobian with respect to the `AppearanceTrackingFactor`'s "pose" variable.
  public let errorVector_H_pose: Tensor<Double>

  /// The Jacobian with respect to the `AppearanceTrackingFactor`'s "latent" variable.
  public let errorVector_H_latent: Tensor<Double>

  /// The IDs of the variables adjacent to this factor.
  public let edges: Variables.Indices

  /// Creates an instance with the given arguments.
  public init(
    error: Patch,
    errorVector_H_pose: Tensor<Double>,
    errorVector_H_latent: Tensor<Double>,
    edges: Variables.Indices
  ) {
    precondition(
      errorVector_H_pose.shape == error.shape + [Pose2.TangentVector.dimension],
      "\(errorVector_H_pose.shape) \(error.shape) \(Pose2.TangentVector.dimension)")
    precondition(
      errorVector_H_latent.shape == error.shape + [LatentCode.dimension],
      "\(errorVector_H_latent.shape) \(error.shape) \(LatentCode.dimension)")
    self.error = error
    self.errorVector_H_pose = errorVector_H_pose
    self.errorVector_H_latent = errorVector_H_latent
    self.edges = edges
  }

  public func error(at x: Variables) -> Double {
    return 0.5 * errorVector(at: x).squaredNorm
  }

  @differentiable
  public func errorVector(at x: Variables) -> Patch {
    errorVector_linearComponent(x) - error
  }

  public func errorVector_linearComponent(_ x: Variables) -> Patch {
    let pose = x.head
    let latent = x.tail.head
    return Patch(
      matmul(errorVector_H_pose, pose.flatTensor.expandingShape(at: 1)).squeezingShape(at: 3)
       + matmul(errorVector_H_latent, latent.flatTensor.expandingShape(at: 1)).squeezingShape(at: 3))
  }

  public func errorVector_linearComponent_adjoint(_ y: Patch) -> Variables {
    let t = y.tensor.reshaped(to: [error.dimension, 1])
    let pose = matmul(
      errorVector_H_pose.reshaped(to: [error.dimension, 3]),
      transposed: true,
      t)
    let latent = matmul(
      errorVector_H_latent.reshaped(to: [error.dimension, LatentCode.dimension]),
      transposed: true,
      t)
    return Tuple2(Vector3(flatTensor: pose), LatentCode(flatTensor: latent))
  }
}
