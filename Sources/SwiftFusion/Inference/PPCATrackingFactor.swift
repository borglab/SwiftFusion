import PenguinStructures
import TensorFlow

/// A factor over a target's pose and appearance in an image.
public struct PPCATrackingFactor: LinearizableFactor2 {
  /// The first adjacent variable, the pose of the target in the image.
  public typealias V0 = Pose2

  /// The second adjacent variable, the PPCA latent code for the appearance of the target.
  public typealias V1 = Vector5

  /// A region cropped from the image.
  public typealias Patch = Tensor28x62x1

  /// The IDs of the variables adjacent to this factor.
  public let edges: Variables.Indices

  /// The image containing the target.
  public let measurement: Tensor<Double>

  /// The PPCA appearance model weight matrix.
  ///
  /// The shape is `Patch.shape + [V1.dimension]`.
  public var W: Tensor<Double>

  /// The PPCA appearance model mean.
  public var mu: Patch

  /// Creates an instance.
  ///
  /// - Requires `W.shape == Patch.shape + [V1.dimension]`.
  public init(_ poseId: TypedID<Pose2>, _ latentId: TypedID<Vector5>, measurement: Tensor<Double>, W: Tensor<Double>, mu: Patch) {
    precondition(W.shape == Patch.shape + [V1.dimension])
    self.edges = Tuple2(poseId, latentId)
    self.measurement = measurement
    self.W = W
    self.mu = mu
  }

  /// Returns the difference between the PPCA generated `Patch` and the `Patch` cropped from
  /// `measurement`.
  @differentiable
  public func errorVector(_ pose: Pose2, _ latent: Vector5) -> Patch.TangentVector {
    let bbox = OrientedBoundingBox(center: pose, rows: Patch.shape[0], cols: Patch.shape[1])
    let generatedAppearance =
      mu + Patch(matmul(W, latent.flatTensor.expandingShape(at: 1)).squeezingShape(at: 2))
    return generatedAppearance - Patch(measurement.patch(at: bbox))
  }

  /// Returns a linear approximation to `self` at `x`.
  public func linearized(at x: Variables) -> LinearizedPPCATrackingFactor {
    let pose = x.head
    let latent = x.tail.head

    let bbox = OrientedBoundingBox(center: pose, rows: Patch.shape[0], cols: Patch.shape[1])
    let generatedAppearance =
      mu + Patch(matmul(W, latent.flatTensor.expandingShape(at: 1)).squeezingShape(at: 2))

    let actualAppearance = measurement.patchWithJacobian(at: bbox)

    return LinearizedPPCATrackingFactor(
      error: Patch(actualAppearance.patch) - generatedAppearance,
      jacobian: Tensor(concatenating: [-actualAppearance.jacobian, W], alongAxis: -1),
      edges: Variables.linearized(edges))
  }
}

/// A linear approximation to `PPCATrackingFactor`, at a certain linearization point.
public struct LinearizedPPCATrackingFactor: GaussianFactor {

  /// The tangent vectors of the `PPCATrackingFactor`'s "pose" and "latent" variables.
  public typealias Variables = Tuple2<Pose2.TangentVector, Vector5>

  /// The error vector at the linearization point.
  public let error: PPCATrackingFactor.Patch

  /// The linear transformation mapping small changes in input variables to small changes in error
  /// around the linearization point.
  ///
  /// The shape is `PPCATrackingFactor.Patch.shape + [Variables.dimension]`.
  public let jacobian: Tensor<Double>

  /// The IDs of the variables adjacent to this factor.
  public let edges: Variables.Indices

  /// Creates an instance with the given `errorVector` and `jacobian`.
  ///
  /// - Requires: `jacobian.shape == PPCATrackingFactor.Patch.shape + [Variables.dimension]`.
  public init(error: PPCATrackingFactor.Patch, jacobian: Tensor<Double>, edges: Variables.Indices) {
    precondition(jacobian.shape == PPCATrackingFactor.Patch.shape + [Variables.dimension])
    self.error = error
    self.jacobian = jacobian
    self.edges = edges
  }

  public func error(at x: Variables) -> Double {
    return 0.5 * errorVector(at: x).squaredNorm
  }

  @differentiable
  public func errorVector(at x: Variables) -> PPCATrackingFactor.Patch {
    errorVector_linearComponent(x) - error
  }

  public func errorVector_linearComponent(_ x: Variables) -> PPCATrackingFactor.Patch {
    PPCATrackingFactor.Patch(
      matmul(jacobian, x.flatTensor.expandingShape(at: 1)).squeezingShape(at: 2))
  }

  public func errorVector_linearComponent_adjoint(_ y: PPCATrackingFactor.Patch) -> Variables {
    Variables(
      flatTensor: matmul(
        jacobian.reshaped(to: [PPCATrackingFactor.Patch.dimension, Variables.dimension]),
        transposed: true,
        y.tensor.reshaped(to: [PPCATrackingFactor.Patch.dimension, 1])
      ).squeezingShape(at: 1)
    )
  }
}
