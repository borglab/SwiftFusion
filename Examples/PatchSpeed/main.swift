import SwiftFusion
import TensorFlow

let rows = 28
let cols = 62
let latentDimension = 5

let image = Tensor<Double>(randomNormal: [500, 500])
let W = Tensor<Double>(randomNormal: [rows * cols, latentDimension])
let mu = Tensor<Double>(randomNormal: [rows, cols])

typealias Jacobian = [(Vector3, Vector5)]


func computeFasterJacobian() -> Jacobian {
    func errorVector(_ center: Pose2) -> Tensor<Double> {
      let bbox = OrientedBoundingBox(center: center, rows: rows, cols: cols)
      //let generated = matmul(W, latent.flatTensor.expandingShape(at: 1)).reshaped(to: [rows, cols])
      return /*generated*/ -image.patch(at: bbox)
    }

    let (value, pb) = valueWithPullback(at: Pose2(100, 100, 0.5), in: errorVector)

    var jacRows: [(Vector3, Vector5)] = []
    let wScalars = W.scalars
    for i in 0..<rows {
      for j in 0..<cols {
        var basisVector = Tensor<Double>(zeros: [rows, cols])
        basisVector[i, j] = Tensor(1)
        let gradCenter = pb(basisVector)
        let wRow = i * cols + j
        jacRows.append((gradCenter, Vector5(wScalars[(5 * wRow)..<(5 * (wRow + 1))])))
      }
    }
    return jacRows
}

let j = computeFasterJacobian()
print(j[0])
