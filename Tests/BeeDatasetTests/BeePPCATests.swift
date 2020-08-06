import BeeDataset
import SwiftFusion
import TensorFlow
import XCTest

import PenguinStructures

/// A factor that specifies a patch on a latent variable.
public struct PPCATrackingFactor: LinearizableFactor2 {
  public typealias Patch = Tensor28x62
  public let edges: Variables.Indices
  public let measured: Tensor<Double>
  public var W: Tensor<Double>
  public var mu: Patch

  public init(_ poseId: TypedID<Pose2>, latentId: TypedID<Vector5>, measured: Tensor<Double>, W: Tensor<Double>, mu: Patch) {
    self.edges = Tuple2(poseId, latentId)
    self.measured = measured
    self.W = W
    self.mu = mu
  }

  // Current pose of the buzz
  public typealias V0 = Pose2
  
  // Latent space
  public typealias V1 = Vector5


  @differentiable
  public func errorVector(_ pose: Pose2, _ latent: Vector5) -> Patch.TangentVector {
    let bbox = OrientedBoundingBox(center: pose, rows: 28, cols: 62)
    return (mu + Patch(matmul(W, latent.flatTensor.expandingShape(at: 1)).squeezingShape(at: 2)) - Patch(measured.expandingShape(at: 2).patch(at: bbox).squeezingShape(at: 2)))
  }
}

func calculatePPCAParams() -> (W: Tensor<Double>, mu: Tensor<Double>) {
  let frames = BeeFrames(sequenceName: "seq4")!
  let obbs = beeOrientedBoundingBoxes(sequenceName: "seq4")!

  let num_samples = 20
  let images_bw = (0..<num_samples).map { frames[$0].patch(at: obbs[$0]).mean(alongAxes: [2]).flattened() }
  let stacked_bw = Tensor(stacking: images_bw).transposed()
  let stacked_mean = stacked_bw.mean(alongAxes: [1])
  let stacked = stacked_bw - stacked_mean
  let (J_s, J_u, _) = stacked.svd(computeUV: true, fullMatrices: false)
  
  let components_taken = 5
  let sigma_2 = J_s[components_taken...].mean()
  let W = matmul(J_u![0..<J_u!.shape[0], 0..<components_taken], (J_s[0..<components_taken] - sigma_2).diagonal()).reshaped(to: [28, 62, components_taken ])
  let patch = frames[0].patch(at: obbs[0]).mean(alongAxes: [2]).squeezingShape(at: 2)
  let W_i = pinv(W.reshaped(to: [62*28, 5]))

  let recon = matmul(W, 
  matmul(
      W_i.reshaped(to: [5, 62 * 28 ]),
      patch.reshaped(to: [62 * 28 , 1]) - stacked_mean
  )
  ).squeezingShape(at: 2) + stacked_mean.reshaped(to: [28, 62])

  print("MSE = \(sqrt((patch - recon).squared().mean()))")
  return (W: W, mu: stacked_mean.reshaped(to: [28, 62]))
}

func getLatentFromPatch(_ patch: Tensor<Double>, W: Tensor<Double>, mu: Tensor<Double>) -> Tensor<Double> {
  let W_i = pinv(W.reshaped(to: [62*28, 5])).reshaped(to: [5, 62 * 28 ])
  return matmul(W_i, (patch.mean(alongAxes: [2]).squeezingShape(at: 2) - mu).reshaped(to: [62 * 28, 1]))
}

final class BeePPCATests: XCTestCase {
  func testPPCA() {
    let frames = BeeFrames(sequenceName: "seq4")!
    let obbs = beeOrientedBoundingBoxes(sequenceName: "seq4")!

    let num_samples = 20
    let images_bw = (0..<num_samples).map { frames[$0].patch(at: obbs[$0]).mean(alongAxes: [2]).flattened() }
    let stacked_bw = Tensor(stacking: images_bw).transposed()
    let stacked_mean = stacked_bw.mean(alongAxes: [1])
    let stacked = stacked_bw - stacked_mean
    let (J_s, J_u, _) = stacked.svd(computeUV: true, fullMatrices: false)
    
    let components_taken = 5
    let sigma_2 = J_s[components_taken...].mean()
    let W = matmul(J_u![0..<J_u!.shape[0], 0..<components_taken], (J_s[0..<components_taken] - sigma_2).diagonal()).reshaped(to: [28, 62, components_taken ])
    let patch = frames[0].patch(at: obbs[0]).mean(alongAxes: [2]).squeezingShape(at: 2)
    let W_i = pinv(W.reshaped(to: [62*28, 5]))

    let recon = matmul(W, 
    matmul(
        W_i.reshaped(to: [5, 62 * 28 ]),
        patch.reshaped(to: [62 * 28 , 1]) - stacked_mean
    )
    ).squeezingShape(at: 2) + stacked_mean.reshaped(to: [28, 62])

    print("MSE = \(sqrt((patch - recon).squared().mean()))")
  }

  func testPPCATracking() {
    let frames = BeeFrames(sequenceName: "seq4")!
    let obbs = beeOrientedBoundingBoxes(sequenceName: "seq4")!
    print(frames.count)
    print(obbs.count)
    let (W, mu) = calculatePPCAParams()
    
    var fg = FactorGraph()
    var v = VariableAssignments()

    let poseId = v.store(obbs[0].center)
    let initialLatent = Vector5(flatTensor: getLatentFromPatch(frames[0].patch(at: obbs[0]), W: W, mu: mu).flattened())
    let latentId = v.store(initialLatent)

    fg.store(PPCATrackingFactor(poseId, latentId: latentId, measured: frames[1].mean(alongAxes: [2]).squeezingShape(at: 2), W: W, mu: Tensor28x62(mu)))
    fg.store(PriorFactor(latentId, initialLatent))
    print("\(fg.linearized(at: v))")

    var optimizer = LM()
    optimizer.verbosity = .TRYLAMBDA
    
    try? optimizer.optimize(graph: fg, initial: &v)

    print("val = \(v[poseId])")
  }
}