import PenguinParallel
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

  /// Weighting of the error
  public var weight: Double

  /// A generative model that produces an appearance from a latent code.
  ///
  /// - Parameters:
  ///   - latentCode: the input to the generative model, which is a tensor with shape
  ///     `[LatentCode.dimension]`.
  /// - Returns:
  ///   - appearance: the output of the generative model, which is a tensor with shape
  ///     `Patch.shape`.
  public typealias GenerativeModel =
    (_ latentCode: Tensor<Double>) -> Tensor<Double>

  /// The generative model that produces an appearance from a latent code.
  public var appearanceModel: GenerativeModel

  /// The Jacobian of a `GenerativeModel`.
  ///
  /// - Parameters:
  ///   - latentCode: the input to the generative model, which is a tensor with shape
  ///     `[LatentCode.dimension]`.
  /// - Returns:
  ///   - jacobian: the Jacobian of the generative model at the given `latentCode`, which is a
  ///     tensor with shape `Patch.shape + [LatentCode.dimension]`.
  public typealias GenerativeModelJacobian = (_ latentCode: Tensor<Double>) -> Tensor<Double>

  /// The Jacobian of `appearanceModel`.
  public var appearanceModelJacobian: GenerativeModelJacobian

  public var regionSize: (Int, Int)

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
    appearanceModel: @escaping GenerativeModel,
    appearanceModelJacobian: @escaping GenerativeModelJacobian,
    weight: Double = 1.0,
    regionSize: (Int, Int)? = nil
  ) {
    self.edges = Tuple2(poseId, latentId)
    self.measurement = measurement
    self.appearanceModel = appearanceModel
    self.appearanceModelJacobian = appearanceModelJacobian
    self.weight = weight

    if let regionSize = regionSize {
      self.regionSize = regionSize
    } else {
      let appearance = appearanceModel(LatentCode.zero.flatTensor)
      self.regionSize = (appearance.shape[0], appearance.shape[1])
    }
  }

  /// Returns the difference between the PPCA generated `Patch` and the `Patch` cropped from
  /// `measurement`.
  @differentiable
  public func errorVector(_ pose: Pose2, _ latent: LatentCode) -> Patch.TangentVector {
    let appearance = appearanceModel(latent.flatTensor) // -> 100x100 image
    let region = OrientedBoundingBox(
      center: pose, rows: regionSize.0, cols: regionSize.1) // -> hxw region
    return Patch(weight * (appearance - measurement.patch(at: region, outputSize: (appearance.shape[0], appearance.shape[1])))) // -> crop out a hxw region and then scale to 100x100
  }

  @derivative(of: errorVector)
  @usableFromInline
  func vjpErrorVector(_ pose: Pose2, _ latent: LatentCode) -> (value: Patch.TangentVector, pullback: (Patch.TangentVector) -> (Pose2.TangentVector, LatentCode)) {
    let lin = self.linearized(at: Tuple2(pose, latent))
    return (
      lin.error,
      { v in
        let r = lin.errorVector_linearComponent_adjoint(v)
        return (r.head, r.tail.head)
      }
    )
  }

  /// Returns a linear approximation to `self` at `x`.
  public func linearized(at x: Variables) -> LinearizedAppearanceTrackingFactor<LatentCode> {
    let pose = x.head
    let latent = x.tail.head
    let generatedAppearance = appearanceModel(latent.flatTensor)
    let generatedAppearance_H_latent = appearanceModelJacobian(latent.flatTensor)
    let region = OrientedBoundingBox(
      center: pose, rows: regionSize.0, cols: regionSize.1)
    let (actualAppearance, actualAppearance_H_pose) = measurement.patchWithJacobian(at: region, outputSize: (generatedAppearance.shape[0], generatedAppearance.shape[1]))
    assert(generatedAppearance_H_latent.shape == generatedAppearance.shape + [LatentCode.dimension])
    return LinearizedAppearanceTrackingFactor<LatentCode>(
      error: Patch(weight * (actualAppearance - generatedAppearance)),
      errorVector_H_pose: -weight * actualAppearance_H_pose,
      errorVector_H_latent: weight * generatedAppearance_H_latent,
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
      .init(ArrayBuffer<LinearizedAppearanceTrackingFactor<LatentCode>>(
        count: factors.count, minimumCapacity: factors.count) { b in
        ComputeThreadPools.local.parallelFor(n: factors.count) { (i, _) in
          let f = factors[factors.index(factors.startIndex, offsetBy: i)]
          (b + i).initialize(to: f.linearized(at: Variables(at: f.edges, in: varsBufs)))
        }
      })
    }
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
    startTimer("linearized appearance")
    defer { stopTimer("linearized appearance") }
    incrementCounter("linearized appearance")
    let pose = x.head
    let latent = x.tail.head
    return Patch(
      matmul(errorVector_H_pose, pose.flatTensor.expandingShape(at: 1)).squeezingShape(at: 3)
       + matmul(errorVector_H_latent, latent.flatTensor.expandingShape(at: 1)).squeezingShape(at: 3))
  }

  public func errorVector_linearComponent_adjoint(_ y: Patch) -> Variables {
    startTimer("linearized appearance transpose")
    defer { stopTimer("linearized appearance transpose") }
    incrementCounter("linearized appearance transpose")
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
