
import BeeDataset
import PenguinStructures
import SwiftFusion
import TensorFlow
import PythonKit
import Foundation

/// Plots a trajectory comparison plot
public func plotTrajectory(track: [Pose2], withGroundTruth expected: [Pose2], on ax: PythonObject,
  withTrackColors predColormap: PythonObject, withGtColors gtColormap: PythonObject) {
  let mpcollections = Python.import("matplotlib.collections")

  let traj_x_gt = expected.map { $0.t.x }
  let traj_y_gt = expected.map { $0.t.y }
  let traj_tensor_gt = Tensor<Double>(stacking: [Tensor(traj_x_gt), Tensor(traj_y_gt)]).transposed()

  let traj_x_pred = track.map { $0.t.x }
  let traj_y_pred = track.map { $0.t.y }
  let traj_tensor_pred = Tensor<Double>(stacking: [Tensor(traj_x_pred), Tensor(traj_y_pred)]).transposed()

  let segments_gt = Tensor(concatenating: [traj_tensor_gt[...(traj_tensor_gt.shape[0]-2)], traj_tensor_gt[1...]], alongAxis: 1)

  let coll_gt = mpcollections.LineCollection(segments_gt.reshaped(to: [segments_gt.shape[0], 2, 2]).makeNumpyArray(), cmap: gtColormap)
  coll_gt.set_array(Tensor<Double>(Array<Double>(traj_x_gt.indices.map { Double($0) })).makeNumpyArray())

  ax.add_collection(coll_gt)
  ax.scatter(traj_x_gt, traj_y_gt)


  let segments_pred = Tensor(concatenating: [traj_tensor_pred[...(traj_tensor_pred.shape[0]-2)], traj_tensor_pred[1...]], alongAxis: 1)

  let coll_pred = mpcollections.LineCollection(segments_pred.reshaped(to: [segments_pred.shape[0], 2, 2]).makeNumpyArray(), cmap: predColormap)
  coll_pred.set_array(Tensor<Double>(Array<Double>(traj_x_pred.indices.map { Double($0) })).makeNumpyArray())
  ax.add_collection(coll_pred)
  ax.scatter(traj_x_pred, traj_y_pred, marker: "x")

  ax.autoscale_view()
  ax.axis("equal")
}

/// Plot the tracking metrics
public func plotOverlap(
  track: [Pose2], withGroundTruth expected: [Pose2], on ax: PythonObject,
  boxSize: (Int, Int) = (40, 70)
) {
  precondition(track.count == expected.count)
  func box(_ p: Pose2) -> OrientedBoundingBox {
    OrientedBoundingBox(center: p, rows: boxSize.0, cols: boxSize.1)
  }
  let metrics = SubsequenceMetrics(groundTruth: expected.map(box), prediction: track.map(box))

  ax.plot(metrics.overlap)
  ax.set_title("Overlap")
}

/// plot Comparison image
public func plotPatchWithGT(frame: Tensor<Float>, actual: Pose2, expected: Pose2) -> (PythonObject, PythonObject) {
  let plt = Python.import("matplotlib.pyplot")

  let extraMarginMultiplier = 2
  let (fig, ax) = plt.subplots(1, 2, figsize: Python.tuple([8, 4])).tuple2
  ax[0].imshow(frame.patch(
    at: OrientedBoundingBox(center: actual, rows: 40 + 20 * extraMarginMultiplier, cols: 70 + 20 * extraMarginMultiplier)
  ).makeNumpyArray() / 255.0)
  ax[0].title.set_text("Prediction")
  ax[1].imshow(frame.patch(
    at: OrientedBoundingBox(center: expected, rows: 40 + 20 * extraMarginMultiplier, cols: 70 + 20 * extraMarginMultiplier)
  ).makeNumpyArray() / 255.0)
  ax[1].title.set_text("Labeling")
  return (fig, ax)
}