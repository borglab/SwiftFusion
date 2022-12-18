import _Differentiation
import BeeDataset
import SwiftFusion
// import TensorFlow
import XCTest

import PenguinStructures

final class BeePPCATests: XCTestCase {
  /// Sanity test that nothing underlying the PPCA algorithm has changed
  func testPPCA() {
    let frames = BeeFrames(sequenceName: "seq4")!
    let obbs = beeOrientedBoundingBoxes(sequenceName: "seq4")!

    let num_samples = 20
    let images_bw = (0..<num_samples).map { frames[$0].patch(at: obbs[$0]).mean(alongAxes: [2]) }
    let stacked_bw = Tensor(stacking: images_bw)
    var ppca = PPCA(latentSize: 5)
    ppca.train(images: stacked_bw)

    let patch = frames[0].patch(at: obbs[0]).mean(alongAxes: [2])
    
    let recon = ppca.decode(ppca.encode(patch))

    XCTAssertEqual(sqrt((patch - recon).squared().mean()).scalar!, 10.049659587270698, accuracy: 1e-5)
  }

  /// Test whether we can actually track a bee with our PPCATrackingFactor
  func testPPCATracking() {
    let frames = BeeFrames(sequenceName: "seq4")!
    let obbs = beeOrientedBoundingBoxes(sequenceName: "seq4")!
    
    let num_samples = 20
    let images_bw = (0..<num_samples).map { frames[$0].patch(at: obbs[$0]).mean(alongAxes: [2]) }
    let stacked_bw = Tensor(stacking: images_bw)
    var ppca = PPCA(latentSize: 5)
    ppca.train(images: stacked_bw)
    
    var fg = FactorGraph()
    var v = VariableAssignments()

    let poseId = v.store(obbs[0].center)
    let initialLatent = Vector5(flatTensor: ppca.encode(frames[0].patch(at: obbs[0]).mean(alongAxes: [2])).flattened())
    let latentId = v.store(initialLatent)

    // Tracking factor on the next frame
    fg.store(AppearanceTrackingFactor(
      poseId, latentId,
      measurement: Tensor<Float>(frames[1].mean(alongAxes: [2])),
      appearanceModel: ppca.decode, appearanceModelJacobian: { _ in ppca.W }))

    // Prior on latent initialized by PPCA decode on the previous frame
    fg.store(PriorFactor(latentId, initialLatent))

    var optimizer = LM()
    optimizer.verbosity = .SILENT
    
    try? optimizer.optimize(graph: fg, initial: &v)

    let expected = Pose2(Rot2(-1.3365823146263909), Vector2(364.59389156740497, 176.17400761774488))

    XCTAssertEqual(expected.localCoordinate(v[poseId]).norm, 0, accuracy: 1e-2)
  }
}
