import BeeDataset
import SwiftFusion
import TensorFlow
import XCTest

extension Tensor where Scalar == Float {
  /// Samples a patch of `self` from `region`, interpolating pixels using `interpolation`.
  ///
  /// - Requires:
  ///   - `self.shape == [width, height, channels]`
  func sample(
    from region: OrientedBoundingBox,
    interpolation: (Tensor<Scalar>, Scalar, Scalar) -> Tensor<Scalar>
  ) -> Tensor<Scalar> {
    precondition(self.shape.count == 3, "image must have shape width x height x channelCount")
    let patchShape: TensorShape = [Int(region.size.x), Int(region.size.y), self.shape[2]]
    var patch = Tensor<Scalar>(zeros: patchShape)
    for i in 0..<patchShape[0] {
      for j in 0..<patchShape[1] {
        let vDest = Vector2(Double(i) + 0.5, Double(j) + 0.5) - 0.5 * region.size
        let vSrc = region.center.t + region.center.rot * vDest
        patch[i, j] = interpolation(self, Float(vSrc.x), Float(vSrc.y))
      }
    }
    return patch
  }
}

/// Returns the bilinear interpolation of point `(x, y)` from `image`.
///
/// - Parameters:
///   - image: a `Tensor` of shape `[width, height, channelCount]". The top left corner is
///     considered coordinate `(0, 0)` and the bottom right corner is considered coordinate
///     `(width, height)`.
@differentiable
func bilinear(
  _ image: Tensor<Float>, x: Float, y: Float
) -> Tensor<Float> {
  precondition(image.shape.count == 3)
  let i = withoutDerivative(at: Int(floor(y - 0.5)))
  let j = withoutDerivative(at: Int(floor(x - 0.5)))
  let p = Float(i) + 1.5 - y
  let q = Float(j) + 1.5 - x

  func pixelOrZero(_ t: Tensor<Float>, _ i: Int, _ j: Int) -> Tensor<Float> {
    if i < 0 {
      return Tensor(zeros: [t.shape[2]])
    }
    if j < 0 {
      return Tensor(zeros: [t.shape[2]])
    }
    if i >= t.shape[0] {
      return Tensor(zeros: [t.shape[2]])
    }
    if j >= t.shape[1] {
      return Tensor(zeros: [t.shape[2]])
    }
    return t[i, j]
  }

  let s1 = pixelOrZero(image, i, j) * p * q
  let s2 = pixelOrZero(image, i + 1, j) * (1 - p) * q
  let s3 = pixelOrZero(image, i, j + 1) * p * (1 - q)
  let s4 = pixelOrZero(image, i + 1, j + 1) * (1 - p) * (1 - q)
  return s1 + s2 + s3 + s4
}

final class BeePPCATests: XCTestCase {
  func testPPCA() {
    let frames = BeeFrames(sequenceName: "seq4")!
    let obbs = beeOrientedBoundingBoxes(sequenceName: "seq4")!

    let num_samples = 20
    let images_bw = (0..<num_samples).map { frames[$0].tensor.sample(from: obbs[$0], interpolation: bilinear).mean(alongAxes: [2]).flattened() }
    let stacked_bw = Tensor(stacking: images_bw).transposed()
    let stacked_mean = stacked_bw.mean(alongAxes: [1])
    let stacked = stacked_bw - stacked_mean
    let (J_s, J_u, _) = stacked.svd(computeUV: true, fullMatrices: false)
    
    let components_taken = 5
    let sigma_2 = J_s[components_taken...].mean()
    let W = matmul(J_u![0..<J_u!.shape[0], 0..<components_taken], J_s[0..<components_taken].subtracting(sigma_2.scalar!).diagonal()).reshaped(to: [62, 28, components_taken ])
    let patch = frames[0].tensor.sample(from: obbs[0], interpolation: bilinear).mean(alongAxes: [2]).squeezingShape(at: 2)
    let W_i = pinv(W.reshaped(to: [62*28, 5]))

    let recon = matmul(W, 
    matmul(
        W_i.reshaped(to: [5, 62 * 28 ]),
        patch.reshaped(to: [62 * 28 , 1]) - stacked_mean
    )
    ).squeezingShape(at: 2) + stacked_mean.reshaped(to: [62,28])

    print("MSE = \(sqrt((patch - recon).squared().mean()))")
  }
}