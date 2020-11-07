// TODO: Deduplicate this with AppearanceTrackingFactor. This is equivalent to an
// AppearanceTrackingFactor with a constant generative model. It's not completely trivial to
// express it that way because AppearanceTrackingFactor assumes that there is a latent code that
// influences the result, and if we ignore the latent code then we get zero derivatives that
// confuse our optimizers.

import PenguinParallel
import PenguinStructures
import TensorFlow

public struct RawPixelTrackingFactor: LinearizableFactor1 {
  public typealias V0 = Pose2
  public let edges: Variables.Indices
  public let measurement: Tensor<Double>
  public let target: Tensor<Double>
  public init(_ poseID: TypedID<Pose2>, measurement: Tensor<Double>, target: Tensor<Double>) {
    self.edges = Tuple1(poseID)
    self.measurement = measurement
    self.target = target
  }

  @differentiable
  public func errorVector(_ pose: Pose2) -> TensorVector {
    let patch = measurement.patch(
      at: OrientedBoundingBox(center: pose, rows: target.shape[0], cols: target.shape[1]))
    return TensorVector(patch - target)
  }

  /// Returns a linear approximation to `self` at `x`.
  public func linearized(at x: Variables) -> CustomLinearization {
    let pose = x.head
    let region = OrientedBoundingBox(center: pose, rows: target.shape[0], cols: target.shape[1])
    let (actualAppearance, actualAppearance_H_pose) = measurement.patchWithJacobian(at: region)
    return CustomLinearization(
      error: TensorVector(-(actualAppearance - target)),
      errorVector_H_pose: actualAppearance_H_pose,
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
      .init(ArrayBuffer<CustomLinearization>(
        count: factors.count, minimumCapacity: factors.count) { b in
        ComputeThreadPools.local.parallelFor(n: factors.count) { (i, _) in
          let f = factors[factors.index(factors.startIndex, offsetBy: i)]
          (b + i).initialize(to: f.linearized(at: Variables(at: f.edges, in: varsBufs)))
        }
      })
    }
  }
}

public struct CustomLinearization: GaussianFactor {
  public typealias Variables = Tuple1<Pose2.TangentVector>

  public typealias Patch = TensorVector

  public let error: Patch

  public let errorVector_H_pose: Tensor<Double>

  public let edges: Variables.Indices

  /// Creates an instance with the given arguments.
  public init(
    error: Patch,
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
  public func errorVector(at x: Variables) -> Patch {
    errorVector_linearComponent(x) - error
  }

  public func errorVector_linearComponent(_ x: Variables) -> Patch {
    let pose = x.head
    return Patch(
      matmul(errorVector_H_pose, pose.flatTensor.expandingShape(at: 1)).squeezingShape(at: 3))
  }

  public func errorVector_linearComponent_adjoint(_ y: Patch) -> Variables {
    let t = y.tensor.reshaped(to: [error.dimension, 1])
    let pose = matmul(
      errorVector_H_pose.reshaped(to: [error.dimension, 3]),
      transposed: true,
      t)
    return Tuple1(Vector3(flatTensor: pose))
  }
}
