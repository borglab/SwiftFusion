import PenguinParallel
import PenguinStructures
import TensorFlow

extension AppearanceModelEncoder {
  @differentiable
  fileprivate func encode<V: FixedSizeVector>(_ image: Tensor<Double>) -> V {
    V(flatTensor: encode(image.expandingShape(at: 0)).squeezingShape(at: 0))
  }
}

/// A factor over a target's pose and appearance in an image.
public struct ProbablisticTrackingFactor<
    LatentCode: FixedSizeVector, Encoder: AppearanceModelEncoder,
    ForegroundModel: GaussianModel, BackgroundModel: GaussianModel
  >: LinearizableFactor2 {
  /// The first adjacent variable, the pose of the target in the image.
  ///
  /// This explicitly specifies `LinearizableFactor2`'s `associatedtype V0`.
  public typealias V0 = Pose2

  /// The second adjacent variable, the latent code for the appearance of the target.
  ///
  /// This explicitly specifies `LinearizableFactor2`'s `associatedtype V1`.
  public typealias V1 = LatentCode

  /// The IDs of the variables adjacent to this factor.
  public let edges: Variables.Indices

  /// The image containing the target.
  public let measurement: Tensor<Double>

  public let encoder: Encoder

  public var patchSize: (Int, Int)

  public var appearanceModelSize: (Int, Int)

  public var foregroundModel: ForegroundModel

  public var backgroundModel: BackgroundModel
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
    encoder: Encoder,
    patchSize: (Int, Int),
    appearanceModelSize: (Int, Int),
    foregroundModel: ForegroundModel,
    backgroundModel: BackgroundModel
  ) {
    self.edges = Tuple2(poseId, latentId)
    self.measurement = measurement
    self.encoder = encoder
    self.patchSize = patchSize
    self.appearanceModelSize = appearanceModelSize
    self.foregroundModel = foregroundModel
    self.backgroundModel = backgroundModel
  }

  @differentiable
  public func errorVector(_ pose: Pose2, _ latent: LatentCode) -> LatentCode {
    let region = OrientedBoundingBox(center: pose, rows: patchSize.0, cols: patchSize.1)
    let patch = measurement.patch(at: region, outputSize: appearanceModelSize)
    return encoder.encode(patch) - latent
  }
}
