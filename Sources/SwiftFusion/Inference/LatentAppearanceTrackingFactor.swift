import PenguinParallel
import PenguinStructures
import TensorFlow

// Same as Encoder. To be used only for an end-to-end classifier
public protocol Classifier {
  @differentiable
  func classify(_ imageBatch: Tensor<Double>) -> Tensor<Double>
}

// To be used as an encoder. 
public protocol Encoder {
  @differentiable
  func encode(_ imageBatch: Tensor<Double>) -> Tensor<Double>
}


public protocol AppearanceModelEncoder : Encoder {
  associatedtype HyperParameters
  init(from imageBatch: Tensor<Double>, given: HyperParameters?)
}

public extension AppearanceModelEncoder {
  /// Extension allows to have a default nil parameter
  init(from imageBatch: Tensor<Double>) {
    self.init(from: imageBatch, given: nil)
  }
  
  @differentiable
  func encode(sample: Tensor<Double>) -> Tensor<Double> {
    encode(sample.expandingShape(at: 0)).squeezingShape(at: 0)
  }
}

extension AppearanceModelEncoder {
  @differentiable
  fileprivate func encode<V: FixedSizeVector>(_ image: Tensor<Double>) -> V {
    V(flatTensor: encode(image.expandingShape(at: 0)).squeezingShape(at: 0))
  }
}

/// A factor over a target's pose and appearance in an image.
public struct LatentAppearanceTrackingFactor<LatentCode: FixedSizeVector, Encoder: AppearanceModelEncoder>: LinearizableFactor2 {
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
    appearanceModelSize: (Int, Int)
  ) {
    self.edges = Tuple2(poseId, latentId)
    self.measurement = measurement
    self.encoder = encoder
    self.patchSize = patchSize
    self.appearanceModelSize = appearanceModelSize
  }

  @differentiable
  public func errorVector(_ pose: Pose2, _ latent: LatentCode) -> LatentCode {
    let region = OrientedBoundingBox(center: pose, rows: patchSize.0, cols: patchSize.1)
    let patch = measurement.patch(at: region, outputSize: appearanceModelSize)
    return encoder.encode(patch) - latent
  }
}
