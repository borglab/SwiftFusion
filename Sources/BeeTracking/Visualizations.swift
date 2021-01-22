
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

/// Plot the tracking metrics
public func plotOverlap(metrics: SubsequenceMetrics, on ax: PythonObject) {
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

/// Calculate the translation error plane (X-Y)
public func errorPlaneTranslation<
  Encoder: AppearanceModelEncoder,
  FGModel: GenerativeDensity,
  BGModel: GenerativeDensity
> (
  frame: Tensor<Float>,
  at: Pose2,
  deltaXs: [Double],
  deltaYs: [Double],
  statistics: FrameStatistics,
  encoder: Encoder,
  foregroundModel: FGModel,
  backgroundModel: BGModel
) -> (fg: Tensor<Double>, bg: Tensor<Double>, e: Tensor<Double>) {
  let targetSize = (40, 70)
  let measurement = statistics.normalized(frame)

  func error(_ pose: Pose2) -> (fg: Double, bg: Double, e: Double) {
    let region = OrientedBoundingBox(center: pose, rows: targetSize.0, cols: targetSize.1)
    let patch = Tensor<Double>(measurement.patch(at: region, outputSize: targetSize))
    let features = encoder.encode(patch.expandingShape(at: 0)).squeezingShape(at: 0)

    let fg_nll = foregroundModel.negativeLogLikelihood(features)
    let bg_nll = backgroundModel.negativeLogLikelihood(features)
    let result = fg_nll - bg_nll

    /// TODO: What is the idiomatic way of avoiding negative probability here?
    return (fg: fg_nll, bg: bg_nll, e: result)
  }

  var fg = Tensor<Double>(zeros: [deltaYs.count, deltaXs.count])
  var bg = Tensor<Double>(zeros: [deltaYs.count, deltaXs.count])
  var errors = Tensor<Double>(zeros: [deltaYs.count, deltaXs.count])
  for (i, dx) in deltaXs.enumerated() {
    for (j, dy) in deltaYs.enumerated() {
      let (fg_nll, bg_nll, e) = error(at * Pose2(dx, dy, 0.0))
      /// x is horiz movement, but is vertical dim in imshow
      fg[j, i] = Tensor(fg_nll)
      bg[j, i] = Tensor(bg_nll)
      errors[j, i] = Tensor(e)
    }
  }
  return (fg, bg, errors)
}

/// Plot the translational error plane
public func plotErrorPlaneTranslation<
  Encoder: AppearanceModelEncoder,
  FGModel: GenerativeDensity,
  BGModel: GenerativeDensity
> (
  frame: Tensor<Float>,
  at pose: Pose2,
  deltaXs: [Double],
  deltaYs: [Double],
  statistics: FrameStatistics,
  encoder: Encoder,
  foregroundModel: FGModel,
  backgroundModel: BGModel,
  normalizeScale: Bool = false
) -> (PythonObject, PythonObject) {
  let plt = Python.import("matplotlib.pyplot")
  let (fg, bg, e) = errorPlaneTranslation(
    frame: frame,
    at: pose,
    deltaXs: deltaXs,
    deltaYs: deltaYs,
    statistics: statistics,
    encoder: encoder,
    foregroundModel: foregroundModel,
    backgroundModel: backgroundModel
  )

  let trans_mins = [e, fg, bg].map { $0.min() }
  let trans_maxs = [e, fg, bg].map { $0.max() }
  let trans_min = Tensor<Double>(trans_mins).min().scalarized()
  let trans_max = Tensor<Double>(trans_maxs).max().scalarized()
  
  let (fig, axs) = plt.subplots(2, 2, figsize: Python.tuple([12, 10])).tuple2
  
  // Plot the image patch
  let img_m = axs[0][0].imshow(frame.patch(
    at: OrientedBoundingBox(center: pose, rows: 40 + 20 * 2, cols: 70 + 20 * 2)
  ).makeNumpyArray() / 255.0, cmap: "gray")
  fig.colorbar(img_m, ax: axs[0][0])
  axs[0][0].title.set_text("Image")
  axs[0][0].set(xlabel: "x displacement", ylabel: "y displacement")

  // Plot the foreground model
  let fg_m = normalizeScale ?
    axs[0][1].imshow(fg.makeNumpyArray(), cmap: "hot", interpolation: "nearest", vmin: trans_min, vmax: trans_max) :
    axs[0][1].imshow(fg.makeNumpyArray(), cmap: "hot", interpolation: "nearest")

  fig.colorbar(fg_m, ax: axs[0][1])
  axs[0][1].title.set_text("Foreground Response")
  axs[0][1].set(xlabel: "x displacement", ylabel: "y displacement")

  // Plot the background model
  let bg_m = normalizeScale ?
    axs[1][0].imshow(bg.makeNumpyArray(), cmap: "hot", interpolation: "nearest", vmin: trans_min, vmax: trans_max):
    axs[1][0].imshow(bg.makeNumpyArray(), cmap: "hot", interpolation: "nearest")

  fig.colorbar(bg_m, ax: axs[1][0])
  axs[1][0].title.set_text("Background Response")
  axs[1][0].set(xlabel: "x displacement", ylabel: "y displacement")

  // Plot the response (error)
  let pcm = normalizeScale ?
    axs[1][1].imshow(e.makeNumpyArray(), cmap: "hot", interpolation: "nearest", vmin: trans_min, vmax: trans_max):
    axs[1][1].imshow(e.makeNumpyArray(), cmap: "hot", interpolation: "nearest")

  fig.colorbar(pcm, ax: axs[1][1])
  axs[1][1].title.set_text("Total Response")
  axs[1][1].set(xlabel: "x displacement", ylabel: "y displacement")

  return (fig, axs)
}

/// Plot the translational error plane for a `TrackingLikelihoodModel`
public func plotErrorPlaneTranslation<
  Encoder: AppearanceModelEncoder,
  FGModel: GenerativeDensity,
  BGModel: GenerativeDensity
> (
  frame: Tensor<Float>,
  at pose: Pose2,
  deltaXs: [Double],
  deltaYs: [Double],
  statistics: FrameStatistics,
  likelihoodModel: TrackingLikelihoodModel<Encoder, FGModel, BGModel>,
  normalizeScale: Bool = false
) -> (PythonObject, PythonObject) {
  plotErrorPlaneTranslation(
    frame: frame,
    at: pose,
    deltaXs: deltaXs,
    deltaYs: deltaYs,
    statistics: statistics,
    encoder: likelihoodModel.encoder,
    foregroundModel: likelihoodModel.foregroundModel,
    backgroundModel: likelihoodModel.backgroundModel,
    normalizeScale: normalizeScale
  )
}

/// Calculate the translational error plane, but for a `TrackingLikelihoodModel`
public func errorPlaneTranslation<
  Encoder: AppearanceModelEncoder,
  FGModel: GenerativeDensity,
  BGModel: GenerativeDensity
> (
  frame: Tensor<Float>,
  at: Pose2,
  deltaXs: [Double],
  deltaYs: [Double],
  statistics: FrameStatistics,
  likelihoodModel: TrackingLikelihoodModel<Encoder, FGModel, BGModel>
) -> (fg: Tensor<Double>, bg: Tensor<Double>, e: Tensor<Double>) {
  errorPlaneTranslation(
    frame: frame, at: at, deltaXs: deltaXs, deltaYs: deltaYs, statistics: statistics,
    encoder: likelihoodModel.encoder, foregroundModel: likelihoodModel.foregroundModel, backgroundModel: likelihoodModel.backgroundModel
  )
}
