// TODO: Deduplicate this with AppearanceTrackingFactor. This is equivalent to an
// AppearanceTrackingFactor with a constant generative model. It's not completely trivial to
// express it that way because AppearanceTrackingFactor assumes that there is a latent code that
// influences the result, and if we ignore the latent code then we get zero derivatives that
// confuse our optimizers.

import PenguinParallel
import PenguinStructures
import TensorFlow

/// A factor that measures the difference between a `target` image patch and the actual image
/// patch at a given pose.
public struct RawPixelTrackingFactor: LinearizableFactor1 {
  /// The adjacent variable, the pose of the target in the image.
  ///
  /// This explicitly specifies `LinearizableFactor2`'s `associatedtype V0`.
  public typealias V0 = Pose2

  /// The IDs of the variables adjacent to this factor.
  public let edges: Variables.Indices

  /// The image containing the target.
  public let measurement: ArrayImage

  /// The pixels of the target.
  public let target: Tensor<Double>

  /// Creates an instance.
  ///
  /// - Parameters:
  ///   - poseId: the id of the adjacent pose variable.
  ///   - measurement: the image containing the target.
  ///   - target: the pixels of the target.
  public init(_ poseID: TypedID<Pose2>, measurement: Tensor<Float>, target: Tensor<Double>) {
    self.edges = Tuple1(poseID)
    self.measurement = ArrayImage(measurement)
    self.target = target
  }

  /// Returns the difference between `target` and the region of `measurement` at `pose`.
  @differentiable
  public func errorVector(_ pose: Pose2) -> TensorVector {
    let patch = measurement.patch(
      at: OrientedBoundingBox(center: pose, rows: target.shape[0], cols: target.shape[1]))
    return TensorVector(Tensor<Double>(patch.tensor) - target)
  }

  /// Returns a linear approximation to `self` at `x`.
  public func linearized(at x: Variables) -> LinearizedRawPixelTrackingFactor {
    let pose = x.head
    let region = OrientedBoundingBox(center: pose, rows: target.shape[0], cols: target.shape[1])
    let (actualAppearance, actualAppearance_H_pose) = measurement.patchWithJacobian(at: region)
    let actualAppearance_H_pose_tensor = Tensor<Double>(Tensor(stacking: [
      actualAppearance_H_pose.dtheta.tensor,
      actualAppearance_H_pose.du.tensor,
      actualAppearance_H_pose.dv.tensor,
    ], alongAxis: -1))
    return LinearizedRawPixelTrackingFactor(
      error: TensorVector(-(Tensor<Double>(actualAppearance.tensor) - target)),
      errorVector_H_pose: actualAppearance_H_pose_tensor,
      edges: Variables.linearized(edges))
  }

  /// Returns the linearizations of `factors` at `x`.
  ///
  /// Note: This causes factor graph linearization to use our custom linearization,
  /// `LinearizedPPCATrackingFactor` instead of the default AD-generated linearization.
  public static func linearized<C: Collection>(_ factors: C, at x: VariableAssignments)
    -> AnyGaussianFactorArrayBuffer where C.Element == Self
  {
    Variables.withBufferBaseAddresses(x) { varsBufs in
      .init(ArrayBuffer<LinearizedRawPixelTrackingFactor>(
        count: factors.count, minimumCapacity: factors.count) { b in
        ComputeThreadPools.local.parallelFor(n: factors.count) { (i, _) in
          let f = factors[factors.index(factors.startIndex, offsetBy: i)]
          (b + i).initialize(to: f.linearized(at: Variables(at: f.edges, in: varsBufs)))
        }
      })
    }
  }
}

/// A linear approximation to `RawPixelTrackingFactor`, at a certain linearization point.
public struct LinearizedRawPixelTrackingFactor: GaussianFactor {
  /// The tangent vector of the `RawPixelTrackingFactor`'s "pose" variable.
  public typealias Variables = Tuple1<Pose2.TangentVector>

  /// The error vector at the linearization point.
  public let error: TensorVector

  /// The Jacobian with respect to the `RawPixelTrackingFactor`'s "pose" variable.
  public let errorVector_H_pose: Tensor<Double>

  /// The ID of the variable adjacent to this factor.
  public let edges: Variables.Indices

  /// Creates an instance with the given arguments.
  public init(
    error: TensorVector,
    errorVector_H_pose: Tensor<Double>,
    edges: Variables.Indices
  ) {
    precondition(
      errorVector_H_pose.shape == error.shape + [Pose2.TangentVector.dimension],
      "\(errorVector_H_pose.shape) \(error.shape) \(Pose2.TangentVector.dimension)")
    self.error = error
    self.errorVector_H_pose = errorVector_H_pose
    self.edges = edges
  }

  public func error(at x: Variables) -> Double {
    return 0.5 * errorVector(at: x).squaredNorm
  }

    @differentiable
  public func errorVector(at x: Variables) -> TensorVector {
    errorVector_linearComponent(x) - error
  }

  public func errorVector_linearComponent(_ x: Variables) -> TensorVector {
    let pose = x.head
    return TensorVector(
      matmul(errorVector_H_pose, pose.flatTensor.expandingShape(at: 1)).squeezingShape(at: 3))
  }

  public func errorVector_linearComponent_adjoint(_ y: TensorVector) -> Variables {
    let t = y.tensor.reshaped(to: [error.dimension, 1])
    let pose = matmul(
      errorVector_H_pose.reshaped(to: [error.dimension, 3]),
      transposed: true,
      t)
    return Tuple1(Vector3(flatTensor: pose))
  }
}
