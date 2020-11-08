import SwiftFusion
import TensorFlow
import Plotly
import ModelSupport
import Foundation

/// A Plotly figure that displays `frame`, with optional `boxes` overlaid.
func plotImagePlotly(
  _ frame: Tensor<Double>, boxes: [(name: String, OrientedBoundingBox)] = [],
  margin: Double = 30, scale: Double = 1
) -> Plotly.Figure {
  let rows = Double(frame.shape[0])
  let cols = Double(frame.shape[1])

  // Axis settings:
  // - no grid
  // - range is the image size
  // - scale is anchored, to preserve image aspect ratio
  // - y axis reversed so that everything is in "(u, v)" coordinates
  let xAx = Layout.XAxis(range: [0, InfoArray(cols)], showGrid: false)
  let yAx = Layout.YAxis(
    autoRange: .reversed, range: [0, InfoArray(rows)], scaleAnchor: .xAxis(xAx), showGrid: false)

  let tmpPath = URL(fileURLWithPath: "tmpForPlotlyDisplay.png")
  ModelSupport.Image(Tensor<Float>(frame)).save(to: tmpPath)
  let imageData = try! "data:image/png;base64," + Data(contentsOf: tmpPath).base64EncodedString()

  return Figure(
    data: [
      // Dummy data because Plotly is confused when there is no data.
      Scatter(
        x: [0, cols], y: [0, rows],
        mode: .markers, marker: Shared.GradientMarker(opacity: 0),
        xAxis: xAx, yAxis: yAx
      )
    ] + boxes.map { box in
      Scatter(
        name: box.name,
        x: box.1.corners.map { $0.x },
        y: box.1.corners.map { $0.y },
        xAxis: xAx,
        yAxis: yAx
      )
    },
    layout: Layout(
      width: cols * scale + 2 * margin,
      height: rows * scale + 2 * margin,
      margin: Layout.Margin(l: margin, r: margin, t: margin, b: margin),
      images: [
        Layout.Image(
          visible: true,
          source: imageData,
          layer: .below,
          xSize: cols, ySize: rows,
          sizing: .stretch,
          x: 0, y: 0, xReference: .xAxis(xAx), yReference: .yAxis(yAx)
        )
      ]
    )
  )
}