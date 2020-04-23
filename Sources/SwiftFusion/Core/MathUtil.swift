//===----------------------------------------------------------------------===//
// Pseudo inverse
//===----------------------------------------------------------------------===//

import TensorFlow

public func pinv(_ m: Tensor<Double>) -> Tensor<Double> {
  let (J_s, J_u, J_v) = m.svd(computeUV: true, fullMatrices: true)
  
  let m = J_v!.shape[1]
  let n = J_u!.shape[0]
  if (m > n) {
    let J_ss = J_s.reciprocal.diagonal().concatenated(with: Tensor<Double>(repeating: 0, shape: [m-n, n]), alongAxis: 0)
    return matmul(matmul(J_v!, J_ss), J_u!.transposed())
  } else if (m < n) {
    let J_ss = J_s.reciprocal.diagonal().concatenated(with: Tensor<Double>(repeating: 0, shape: [m, n-m]), alongAxis: 1)
    return matmul(matmul(J_v!, J_ss), J_u!.transposed())
  } else {
    let J_ss = J_s.reciprocal.diagonal()
    return matmul(matmul(J_v!, J_ss), J_u!.transposed())
  }
}
