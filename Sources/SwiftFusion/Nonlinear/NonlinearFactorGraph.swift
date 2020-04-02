import TensorFlow

public struct NonlinearFactorGraph {
  var factor1s: [Factor1] = []
  var factor2s: [Factor2] = []

  var factorCount: Int { factor1s.count + factor2s.count }

  public init() {}

  public mutating func add(
    _ value1Index: Int, _ error: @escaping @differentiable (Pose2) -> Pose2, _ noise: NoiseModel
  ) {
    factor1s.append(Factor1(value1Index: value1Index, error: error, noise: noise))
  }

  public mutating func add(
    _ value1Index: Int, _ value2Index: Int,
    _ error: @escaping @differentiable (Pose2, Pose2) -> Pose2,
    _ noise: NoiseModel
  ) {
    factor2s.append(Factor2(
      value1Index: value1Index, value2Index: value2Index, error: error, noise: noise))
  }

  public func loss(at point: [Pose2]) -> Double {
    factor1s.map { $0.noise.loss($0.error(point[$0.value1Index])) }.reduce(0, +) +
      factor2s.map { $0.noise.loss($0.error(point[$0.value1Index], point[$0.value2Index])) }.reduce(0, +)
  }

  /// Returns an nx1 matrix (a vector) whose entries are the factors' errors at `point`.
  public func errors(at point: [Pose2]) -> Tensor<Double> {
    var vector = Tensor<Double>(zeros: [3 * factorCount, 1])
    var currentFactorIndex: Int = 0
    for factor in factor1s {
      let error = factor.error(point[factor.value1Index])
      vector[3 * currentFactorIndex, 0] = Tensor(error.rot.theta)
      vector[3 * currentFactorIndex + 1, 0] = Tensor(error.t.x)
      vector[3 * currentFactorIndex + 2, 0] = Tensor(error.t.y)
      currentFactorIndex += 1
    }
    for factor in factor2s {
      let error = factor.error(point[factor.value1Index], point[factor.value2Index])
      vector[3 * currentFactorIndex, 0] = Tensor(error.rot.theta)
      vector[3 * currentFactorIndex + 1, 0] = Tensor(error.t.x)
      vector[3 * currentFactorIndex + 2, 0] = Tensor(error.t.y)
      currentFactorIndex += 1
    }
    return vector
  }

  /// Returns a matrix representing a linear approximation of changes in factor errors as a
  /// function of changes in `point`.
  ///
  /// Note: Currently returns a dense representation of the matrix, even though the matrix values
  /// are actually very sparse.
  public func linearize(at point: [Pose2]) -> Tensor<Double> {
    var matrix = Tensor<Double>(zeros: [3 * factorCount, 3 * point.count])
    var currentFactorIndex: Int = 0
    for factor in factor1s {
      let pb = pullback(at: point[factor.value1Index], in: factor.error)
      for (basisVectorIndex, basisVector) in Pose2.tangentStandardBasis.enumerated() {
        let rowValue = pb(basisVector)
        let rowIndex = 3 * currentFactorIndex + basisVectorIndex
        matrix[rowIndex, 3 * factor.value1Index] = Tensor(rowValue.x)
        matrix[rowIndex, 3 * factor.value1Index + 1] = Tensor(rowValue.y)
        matrix[rowIndex, 3 * factor.value1Index + 2] = Tensor(rowValue.z)
      }
      currentFactorIndex += 1
    }
    for factor in factor2s {
      let pb = pullback(at: point[factor.value1Index], point[factor.value2Index], in: factor.error)
      for (basisVectorIndex, basisVector) in Pose2.tangentStandardBasis.enumerated() {
        let (rowValue1, rowValue2) = pb(basisVector)
        let rowIndex = 3 * currentFactorIndex + basisVectorIndex
        matrix[rowIndex, 3 * factor.value1Index] = Tensor(rowValue1.x)
        matrix[rowIndex, 3 * factor.value1Index + 1] = Tensor(rowValue1.y)
        matrix[rowIndex, 3 * factor.value1Index + 2] = Tensor(rowValue1.z)
        matrix[rowIndex, 3 * factor.value2Index] = Tensor(rowValue2.x)
        matrix[rowIndex, 3 * factor.value2Index + 1] = Tensor(rowValue2.y)
        matrix[rowIndex, 3 * factor.value2Index + 2] = Tensor(rowValue2.z)
      }
      currentFactorIndex += 1
    }
    return matrix
  }
}

public struct NoiseModel {
  public let sigmaX, sigmaY, sigmaRot: Double
  public func loss(_ error: Pose2) -> Double {
    sigmaX * error.t.x * error.t.x +
      sigmaY * error.t.y * error.t.y +
      sigmaRot * error.rot.theta * error.rot.theta
  }
  public init(_ sigmaX: Double, _ sigmaY: Double, _ sigmaRot: Double) {
    self.sigmaX = sigmaX
    self.sigmaY = sigmaY
    self.sigmaRot = sigmaRot
  }
}

struct Factor1 {
  let value1Index: Int
  let error: @differentiable (Pose2) -> Pose2
  let noise: NoiseModel
}

struct Factor2 {
  let value1Index: Int
  let value2Index: Int
  let error: @differentiable (Pose2, Pose2) -> Pose2
  let noise: NoiseModel
}
