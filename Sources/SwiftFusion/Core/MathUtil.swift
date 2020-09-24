//===----------------------------------------------------------------------===//
// Pseudo inverse
//===----------------------------------------------------------------------===//

import TensorFlow

/// Calculate the pseudo-inverse of a matrix
/// Input: [M, N]
/// Output: [N, M]
public func pinv<Scalar: TensorFlowFloatingPoint>(_ m: Tensor<Scalar>) -> Tensor<Scalar> {
  precondition(m.rank == 2, "Wrong input dimension for pinv()")
  
  let (J_s, J_u, J_v) = m.svd(computeUV: true, fullMatrices: true)
  
  let m = J_v!.shape[1]
  let n = J_u!.shape[0]
  if (m > n) {
    let J_ss = J_s.reciprocal.diagonal().concatenated(with: Tensor<Scalar>(repeating: 0, shape: [m-n, n]), alongAxis: 0)
    return matmul(matmul(J_v!, J_ss), J_u!.transposed())
  } else if (m < n) {
    let J_ss = J_s.reciprocal.diagonal().concatenated(with: Tensor<Scalar>(repeating: 0, shape: [m, n-m]), alongAxis: 1)
    return matmul(matmul(J_v!, J_ss), J_u!.transposed())
  } else {
    let J_ss = J_s.reciprocal.diagonal()
    return matmul(matmul(J_v!, J_ss), J_u!.transposed())
  }
}
