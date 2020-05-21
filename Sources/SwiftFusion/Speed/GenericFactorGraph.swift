// MARK: - Factors

struct MyPriorFactor: Factor1 {
  typealias InputIDs = VariableIDs1<Pose2>
  var inputId1: SimpleTypedID<Pose2>

  let prior: Pose2

  init(_ inputId1: SimpleTypedID<Pose2>, _ prior: Pose2) {
    self.inputId1 = inputId1
    self.prior = prior
  }

  @differentiable(wrt: value)
  func errorVector(at value: Pose2) -> Pose2.TangentVector {
    return prior.localCoordinate(value)
  }

  func linearized(at value: Pose2) -> JacobianFactor1<Matrix3x3> {
    return JacobianFactor1(
      linearizing: self.errorVector,
      at: value,
      inputId1: TypedID(inputId1.perTypeID),
      makeJacobian: makeJacobian
    )
  }
}

struct MyBetweenFactor: Factor2 {
  typealias InputIDs = VariableIDs2<Pose2, Pose2>
  var inputId1: SimpleTypedID<Pose2>
  var inputId2: SimpleTypedID<Pose2>

  typealias ErrorVector = Vector3

  let difference: Pose2

  init(_ inputId1: SimpleTypedID<Pose2>, _ inputId2: SimpleTypedID<Pose2>, _ difference: Pose2) {
    self.inputId1 = inputId1
    self.inputId2 = inputId2
    self.difference = difference
  }

  @differentiable(wrt: (value1, value2))
  func errorVector(at value1: Pose2, _ value2: Pose2) -> Vector3 {
    let actualDifference = between(value1, value2)
    return difference.localCoordinate(actualDifference)
  }

  typealias LinearizedFactor = JacobianFactor2<Matrix3x3, Matrix3x3>

  func linearized(at value1: Pose2, _ value2: Pose2) -> JacobianFactor2<Matrix3x3, Matrix3x3> {
    return JacobianFactor2(
      linearizing: self.errorVector,
      at: value1, value2,
      inputId1: TypedID(inputId1.perTypeID),
      inputId2: TypedID(inputId2.perTypeID),
      makeJacobian: makeJacobian
    )
  }
}

struct JacobianFactor1<Jacobian: LinearFunction>: Factor1 {
  typealias InputIDs = VariableIDs1<Jacobian.Input>
  let inputId1: SimpleTypedID<Jacobian.Input>

  let jacobian: Jacobian
  let error: Jacobian.Output

  init<Input: Differentiable>(
    linearizing f: @differentiable (Input) -> Jacobian.Output,
    at values: Input,
    inputId1: SimpleTypedID<Jacobian.Input>,
    makeJacobian: ((Jacobian.Output) -> (Input.TangentVector)) -> Jacobian
  ) where Input.TangentVector == Jacobian.Input {
    let (error, errorPullback) = valueWithPullback(at: values, in: f)
    self.error = error
    self.jacobian = makeJacobian(errorPullback)
    self.inputId1 = inputId1
  }

  func errorVector(at values: Jacobian.Input) -> Jacobian.Output {
    return applyLinearForward(values) + error
  }

  func linearized(at values: Jacobian.Input) -> Self {
    return self
  }

  func applyLinearForward(_ x: Jacobian.Input) -> Jacobian.Output {
    return jacobian.forward(x)
  }

  func applyLinearTranspose(_ y: Jacobian.Output, into result: inout Jacobian.Input) {
    result += jacobian.transpose(y)
  }
}

struct JacobianFactor2<Jacobian1: LinearFunction, Jacobian2: LinearFunction>: Factor2 where Jacobian1.Output == Jacobian2.Output {
  typealias InputIDs = VariableIDs2<Jacobian1.Input, Jacobian2.Input>
  let inputId1: SimpleTypedID<Jacobian1.Input>
  let inputId2: SimpleTypedID<Jacobian2.Input>

  typealias ErrorVector = Jacobian1.Output
  typealias LinearizedFactor = Self

  let jacobian1: Jacobian1
  let jacobian2: Jacobian2
  let error: Jacobian1.Output

  init<Input1: Differentiable, Input2: Differentiable>(
    linearizing f: @differentiable (Input1, Input2) -> Jacobian1.Output,
    at value1: Input1, _ value2: Input2,
    inputId1: SimpleTypedID<Jacobian1.Input>,
    inputId2: SimpleTypedID<Jacobian2.Input>,
    makeJacobian: ((Jacobian1.Output) -> (Input1.TangentVector, Input2.TangentVector)) -> (Jacobian1, Jacobian2)
  ) where Input1.TangentVector == Jacobian1.Input, Input2.TangentVector == Jacobian2.Input {
    let (error, errorPullback) = valueWithPullback(at: value1, value2, in: f)
    self.error = error
    let (jacobian1, jacobian2) = makeJacobian(errorPullback)
    self.jacobian1 = jacobian1
    self.jacobian2 = jacobian2
    self.inputId1 = inputId1
    self.inputId2 = inputId2
  }

  func errorVector(at value1: Jacobian1.Input, _ value2: Jacobian2.Input) -> Jacobian1.Output {
    return applyLinearForward(value1, value2) + error
  }

  func linearized(at value1: Jacobian1.Input, _ value2: Jacobian2.Input) -> Self {
    return self
  }

  func applyLinearForward(_ x1: Jacobian1.Input, _ x2: Jacobian2.Input) -> Jacobian1.Output {
    return jacobian1.forward(x1) + jacobian2.forward(x2)
  }

  func applyLinearTranspose(_ y: Jacobian1.Output, into result1: inout Jacobian1.Input, _ result2: inout Jacobian2.Input) {
    result1 += jacobian1.transpose(y)
    result2 += jacobian2.transpose(y)
  }
}

// struct MyJacobianFactor2<Jacobian1: LinearFunction, Jacobian2: LinearFunction>: Factor1
//   where Jacobian1.Output == Jacobian2.Output
// {
//   typealias InputIDs = VariableIDs1<Jacobian1.Input, Jacobian2.Input>
//   let inputId1: SimpleTypedID<InputVector>
// 
//   typealias ErrorVector = Jacobian.Output
// 
//   let jacobian: Jacobian
//   let error: ErrorVector
// 
//   init<Input: Differentiable>(
//     linearizing f: @differentiable (Input) -> ErrorVector,
//     at values: Input,
//     inputId1: SimpleTypedID<InputVector>
//   ) where Input.TangentVector == InputVector {
//     let (error, errorPullback) = valueWithPullback(at: values, in: f)
//     self.error = error
//     self.jacobian = Jacobian(transposing: errorPullback)
//     self.inputId1 = inputId1
//   }
// 
//   func errorVector(at values: InputVector) -> ErrorVector {
//     return applyLinearForward(values) + error
//   }
// 
//   func linearized(at values: InputVector) -> Self {
//     return self
//   }
// 
//   func applyLinearForward(_ x: InputVector) -> ErrorVector {
//     return jacobian.forward(x)
//   }
// 
//   func applyLinearTranspose(_ y: ErrorVector, into result: inout InputVector) {
//     result += jacobian.transpose(y)
//   }
// }

// MARK: - Generic FactorGraph

struct FactorGraph {
  var factors: PackedStorage = PackedStorage()

  // tagged by factor type
  var errorZero: PackedStorage = PackedStorage()

  static func += <T: AnyFactor>(_ graph: inout Self, _ factor: T) {
    _ = graph.factors.storeFactor(factor)
    _ = graph.errorZero.storeEuclideanVector(T.ErrorVector.zero, tag: ObjectIdentifier(T.self))
  }

  func error(at values: PackedStorage) -> Double {
    return factors.error(at: values)
  }

  func linearized(at values: PackedStorage) -> MyGaussianFactorGraph {
    var linearized = FactorGraph()
    factors.linearize(at: values, &linearized)
    return MyGaussianFactorGraph(graph: linearized, inputZero: values.zeroTangentVector)
  }
}

struct MyGaussianFactorGraph {
  var graph: FactorGraph
  var inputZero: PackedStorage
}

extension MyGaussianFactorGraph: GaussianFactor {
  func errorVector(_ x: PackedStorage) -> PackedStorage {
    var result = graph.errorZero
    graph.factors.errorVector(at: x, &result)
    return result
  }
  func applyLinearForward(_ x: PackedStorage) -> PackedStorage {
    var result = graph.errorZero
    graph.factors.applyLinearForward(x, &result)
    return result
  }
  func applyLinearTranspose(_ y: PackedStorage) -> PackedStorage {
    var result = inputZero
    graph.factors.applyLinearTranspose(y, &result)
    return result
  }
}

public func runGenericFactorGraph() {
  var values = PackedStorage()
  let variable1 = values.storeDifferentiable(Pose2())
  //let variable2 = values.storeDifferentiable(Pose2())

  var graph = FactorGraph()
  graph += MyPriorFactor(variable1, Pose2(1, 0, 0))
  //graph += ((variable1, variable2), MyBetweenFactor(Pose2(1, 0, 1)))

  print(graph.error(at: values))
  for _ in 0..<5 {
    let linearized = graph.linearized(at: values)
    let optimizer = CGLS(precision: 1e-6, max_iteration: 5)
    var dx = linearized.inputZero
    optimizer.optimize(linearized, initial: &dx)
    values.move(along: dx)
    print(values[variable1]) //, values[variable2])
  }
  print(graph.error(at: values))
}

public func runGenericFactorGraphBenchmark() {
  let intelPoseSLAMFactorGraph = try! MyG2OFactorGraph(fromG2O: try! cachedDataset("input_INTEL_g2o.txt"))
  var graph = intelPoseSLAMFactorGraph.graph
  var values = intelPoseSLAMFactorGraph.initialGuess
  print(graph.error(at: values))
  graph += MyPriorFactor(intelPoseSLAMFactorGraph.variableIDs[0]!, Pose2(0, 0, 0))
  for _ in 0..<10 {
    let gfg = graph.linearized(at: values)
    let optimizer = CGLS(precision: 0, max_iteration: 500)
    var dx = gfg.inputZero
    optimizer.optimize(gfg, initial: &dx)
    values.move(along: dx)
    print(graph.error(at: values))
  }
}

/// Builds an initial guess and a factor graph from a g2o file.
struct MyG2OFactorGraph: G2OReader {
  /// The initial guess.
  var initialGuess: PackedStorage = PackedStorage()

  /// The factor graph representing the measurements.
  var graph: FactorGraph = FactorGraph()

  var variableIDs: [Int: SimpleTypedID<Pose2>] = [:]

  public mutating func addInitialGuess(index: Int, pose: Pose2) {
    variableIDs[index] = initialGuess.storeDifferentiable(pose)
  }

  public mutating func addMeasurement(frameIndex: Int, measuredIndex: Int, pose: Pose2) {
    graph += MyBetweenFactor(variableIDs[frameIndex]!, variableIDs[measuredIndex]!, pose)
  }
}


// // Here is some psuedocode.
// 
// struct DiscreteTransitionFactor: MyFactor {
//   let probability: [[Double]]
// 
//   init(_ probability: [[Double]]) {
//     self.probability = probability
//   }
// 
//   func error(at start: Int, end: Int) -> Double {
//     return -log(probability[start][end])
//   }
// }
// 
// struct SLDSFactor: MyFactor {
//   let movementModels: [Pose2]
// 
//   init(_ movementModels: [Pose2]) {
//     self.movementModels = movementModels
//   }
// 
//   @differentiable(wrt: start, end)
//   func errorVector(at label: Int, _ start: Pose2, _ end: Pose2) -> Vector3 {
//     let actualDifference = between(start, end)
//     return movementModels[label].localCoordinate(actualDifference)
//   }
// 
//   func linearized(at label: Int, _ start: Pose2, _ end: Pose2) -> JacobianFactor<Matrix3x6> {
//     return JacobianFactor(
//       linearizing: { self.errorVector(movementModelIndex, $0, $1) },
//       at: start, end
//     )
//   }
// }
// 
// func doBeeTracking() {
//   let label0: Int = 0
// 
//   let movementModels: [Pose2] = loadMovementModels()
//   let transitionProbabilities: [[Double]] = loadTransitionProbabilities()
// 
//   let observedPositions: [Pose2] = loadObservedPositions()
// 
//   // Create initial guess.
//   var variables = PackedStorage()
//   var labelVariables = (0..<(observedPositions.count - 1)).map { variables.store(label0) }
//   var positionVariables = observedPositions.map { variables.storeDifferentiable($0) }
// 
//   // Create factor graph.
//   var graph = FactorGraph()
//   for t in 0..<(observedPositions.count - 1) {
//     graph += (
//       (labelVariables[t], labelVariables[t + 1]),
//       DiscreteTransitionFactor(transitionProbabilities)
//     )
//     graph += (
//       (labelVariables[t], positionVariables[t], positionVariables[t + 1]),
//       SLDSFactor(movementModels)
//     )
//   }
//   for t in 0..<observedPositions.count {
//     graph += (positionVariables[t], PriorFactor(observedPositions[t]))
//   }
// 
//   // 
// }
