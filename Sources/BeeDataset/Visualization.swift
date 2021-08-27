// // Copyright 2020 The SwiftFusion Authors. All Rights Reserved.
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //     http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.

// import SwiftFusion
// import TensorFlow
// // import Plotly
// import ModelSupport
// import Foundation

// /// Creates a Plotly figure that displays `frame`, with optional `boxes` overlaid on
// /// them.
// public func plot<Scalar: TensorFlowFloatingPoint>(
//   _ frame: Tensor<Scalar>, boxes: [(name: String, OrientedBoundingBox)] = [],
//   margin: Double = 30, scale: Double = 1
// ) -> Plotly.Figure {
//   let rows = Double(frame.shape[0])
//   let cols = Double(frame.shape[1])

//   // Axis settings:
//   // - no grid
//   // - range is the image size
//   // - scale is anchored, to preserve image aspect ratio
//   // - y axis reversed so that everything is in "(u, v)" coordinates
//   let xAx = Layout.XAxis(range: [0, InfoArray(cols)], showGrid: false)
//   let yAx = Layout.YAxis(
//     autoRange: .reversed, range: [0, InfoArray(rows)], scaleAnchor: .xAxis(xAx), showGrid: false)

//   let tmpPath = URL(fileURLWithPath: "tmpForPlotlyDisplay.png")
//   ModelSupport.Image(Tensor<Float>(frame)).save(to: tmpPath)
//   let imageData = try! "data:image/png;base64," + Data(contentsOf: tmpPath).base64EncodedString()

//   return Figure(
//     data: [
//       // Dummy data because Plotly is confused when there is no data.
//       Scatter(
//         x: [0, cols], y: [0, rows],
//         mode: .markers, marker: Shared.GradientMarker(opacity: 0),
//         xAxis: xAx, yAxis: yAx
//       )
//     ] + boxes.map { box in
//       Scatter(
//         name: box.name,
//         x: box.1.corners.map { $0.x },
//         y: box.1.corners.map { $0.y },
//         xAxis: xAx,
//         yAxis: yAx
//       )
//     },
//     layout: Layout(
//       width: cols * scale + 2 * margin,
//       height: rows * scale + 2 * margin,
//       margin: Layout.Margin(l: margin, r: margin, t: margin, b: margin),
//       images: [
//         Layout.Image(
//           visible: true,
//           source: imageData,
//           layer: .below,
//           xSize: cols, ySize: rows,
//           sizing: .stretch,
//           x: 0, y: 0, xReference: .xAxis(xAx), yReference: .yAxis(yAx)
//         )
//       ]
//     )
//   )
// }