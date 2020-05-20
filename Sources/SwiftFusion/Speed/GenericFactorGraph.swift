// MARK: - Factors

struct MyPriorFactor: MyFactor {
  typealias Input = FactorInput1<Pose2>

  let prior: Pose2

  init(_ prior: Pose2) {
    self.prior = prior
  }

  @differentiable(wrt: value)
  func errorVector(at value: FactorInput1<Pose2>) -> Pose2.TangentVector {
    return prior.localCoordinate(value.value1)
  }

  func linearized(at value: FactorInput1<Pose2>) ->
    MyJacobianFactor<FactorVectorInput1<Vector3>, Vector3, Matrix3x3>
  {
    return MyJacobianFactor(linearizing: self.errorVector, at: value)
  }

  func linearized(id: Input.InputID) -> LinearizedFactor.Input.InputID {
    return TypedID(id.perTypeID)
  }
}

struct MyBetweenFactor: MyFactor {
  typealias Input = FactorInput2<Pose2, Pose2>

  let difference: Pose2

  init(_ difference: Pose2) {
    self.difference = difference
  }

  @differentiable(wrt: value)
  func errorVector(at value: FactorInput2<Pose2, Pose2>) -> Vector3 {
    let actualDifference = between(value.value1, value.value2)
    return difference.localCoordinate(actualDifference)
  }

  func linearized(at value: FactorInput2<Pose2, Pose2>) ->
    MyJacobianFactor<FactorVectorInput2<Vector3, Vector3>, Vector3, Matrix3x6>
  {
    return MyJacobianFactor(linearizing: self.errorVector, at: value)
  }

  func linearized(id: Input.InputID) -> LinearizedFactor.Input.InputID {
    return (TypedID(id.0.perTypeID), TypedID(id.1.perTypeID))
  }
}

struct MyJacobianFactor<
  InputVector: EuclideanVectorSpace & VectorConvertible & FactorInputProtocol,
  ErrorVector: EuclideanVectorSpace & VectorConvertible & TangentStandardBasis,
  Jacobian: LinearFunction
>: MyFactor where Jacobian.Input == InputVector, Jacobian.Output == ErrorVector {

  typealias Input = InputVector

  let jacobian: Jacobian
  let error: ErrorVector

  init<Input: Differentiable>(
    linearizing f: @differentiable (Input) -> ErrorVector,
    at values: Input
  ) where Input.TangentVector == InputVector {
    let (error, errorPullback) = valueWithPullback(at: values, in: f)
    self.error = error
    self.jacobian = Jacobian(transposing: errorPullback)
  }

  func errorVector(at values: InputVector) -> ErrorVector {
    return applyLinearForward(values) + error
  }

  func linearized(at values: InputVector) -> Self {
    return self
  }

  func linearized(id: Input.InputID) -> LinearizedFactor.Input.InputID {
    return id
  }

  func applyLinearForward(_ x: InputVector) -> ErrorVector {
    return jacobian.forward(x)
  }

  func applyLinearTranspose(_ y: ErrorVector) -> InputVector {
    return jacobian.transpose(y)
  }
}

// MARK: - Generic FactorGraph

struct FactorGraph {
  var factors: PackedStorage = PackedStorage()

  // tagged by factor type
  var factorInputs: PackedStorage = PackedStorage()

  // tagged by factor type
  var errorZero: PackedStorage = PackedStorage()

  static func += <T: MyFactor>(_ lhs: inout Self, _ rhs: (T.Input.InputID, T)) {
    let (factorInput, factor) = rhs
    _ = lhs.factors.storeFactor(factor)
    _ = lhs.factorInputs.store(factorInput, tag: ObjectIdentifier(T.self))
    _ = lhs.errorZero.storeEuclideanVector(T.ErrorVector.zero, tag: ObjectIdentifier(T.self))
  }

  func error(at values: PackedStorage) -> Double {
    return factors.error(at: values, factorInputs)
  }

  func linearized(at values: PackedStorage) -> MyGaussianFactorGraph {
    var linearized = FactorGraph()
    factors.linearize(at: values, &linearized, factorInputs)
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
    graph.factors.errorVector(at: x, &result, graph.factorInputs)
    return result
  }
  func applyLinearForward(_ x: PackedStorage) -> PackedStorage {
    var result = graph.errorZero
    graph.factors.applyLinearForward(x, &result, graph.factorInputs)
    return result
  }
  func applyLinearTranspose(_ y: PackedStorage) -> PackedStorage {
    var result = inputZero
    graph.factors.applyLinearTranspose(y, &result, graph.factorInputs)
    return result
  }
}

public func runGenericFactorGraph() {
  var values = PackedStorage()
  let variable1 = values.storeDifferentiable(Pose2())
  let variable2 = values.storeDifferentiable(Pose2())

  var graph = FactorGraph()
  graph += (variable1, MyPriorFactor(Pose2(1, 0, 0)))
  graph += ((variable1, variable2), MyBetweenFactor(Pose2(1, 0, 1)))

  print(graph.error(at: values))
  for _ in 0..<5 {
    let linearized = graph.linearized(at: values)
    let optimizer = CGLS(precision: 1e-6, max_iteration: 500)
    var dx = linearized.inputZero
    optimizer.optimize(linearized, initial: &dx)
    values.move(along: dx)
    print(values[variable1], values[variable2])
  }
  print(graph.error(at: values))
}

public func runGenericFactorGraphBenchmark() {
  let intelPoseSLAMFactorGraph = try! MyG2OFactorGraph(fromG2O: try! cachedDataset("input_INTEL_g2o.txt"))
  var graph = intelPoseSLAMFactorGraph.graph
  var values = intelPoseSLAMFactorGraph.initialGuess
  print(graph.error(at: values))
  graph += (intelPoseSLAMFactorGraph.variableIDs[0]!, MyPriorFactor(Pose2(0, 0, 0)))
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
    graph += ((variableIDs[frameIndex]!, variableIDs[measuredIndex]!), MyBetweenFactor(pose))
  }
}

#if false

// Here is some psuedocode.

struct DiscreteTransitionFactor: MyFactor {
  let probability: [[Double]]

  init(_ probability: [[Double]]) {
    self.probability = probability
  }

  func error(at start: Int, end: Int) -> Double {
    return -log(probability[start][end])
  }
}

struct SLDSFactor: MyFactor {
  let movementModels: [Pose2]

  init(_ movementModels: [Pose2]) {
    self.movementModels = movementModels
  }

  @differentiable(wrt: start, end)
  func errorVector(at label: Int, _ start: Pose2, _ end: Pose2) -> Vector3 {
    let actualDifference = between(start, end)
    return movementModels[label].localCoordinate(actualDifference)
  }

  func linearized(at label: Int, _ start: Pose2, _ end: Pose2) -> JacobianFactor<Matrix3x6> {
    return JacobianFactor(
      linearizing: { self.errorVector(movementModelIndex, $0, $1) },
      at: start, end
    )
  }
}

func doBeeTracking() {
  let label0: Int = 0

  let movementModels: [Pose2] = loadMovementModels()
  let transitionProbabilities: [[Double]] = loadTransitionProbabilities()

  let observedPositions: [Pose2] = loadObservedPositions()

  // Create initial guess.
  var variables = PackedStorage()
  var labelVariables = (0..<(observedPositions.count - 1)).map { variables.store(label0) }
  var positionVariables = observedPositions.map { variables.storeDifferentiable($0) }

  // Create factor graph.
  var graph = FactorGraph()
  for t in 0..<(observedPositions.count - 1) {
    graph += (
      (labelVariables[t], labelVariables[t + 1]),
      DiscreteTransitionFactor(transitionProbabilities)
    )
    graph += (
      (labelVariables[t], positionVariables[t], positionVariables[t + 1]),
      SLDSFactor(movementModels)
    )
  }
  for t in 0..<observedPositions.count {
    graph += (positionVariables[t], PriorFactor(observedPositions[t]))
  }

  // 
}

#endif
