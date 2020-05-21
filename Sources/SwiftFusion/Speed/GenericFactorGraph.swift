import Foundation
import TensorFlow

// MARK: - Factors

struct DemoBetweenFactor: DifferentiableFactor2 {
  typealias InputIDs = VariableIDs2<Pose2, Pose2>
  var inputId1: SimpleTypedID<Pose2>
  var inputId2: SimpleTypedID<Pose2>

  let difference: Pose2

  init(_ inputId1: SimpleTypedID<Pose2>, _ inputId2: SimpleTypedID<Pose2>, _ difference: Pose2) {
    self.inputId1 = inputId1
    self.inputId2 = inputId2
    self.difference = difference
  }

  typealias ErrorVector = Vector3

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

struct JacobianFactor1<Jacobian: LinearFunction>: DifferentiableFactor1 {
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

struct JacobianFactor2<Jacobian1: LinearFunction, Jacobian2: LinearFunction>: DifferentiableFactor2 where Jacobian1.Output == Jacobian2.Output {
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

struct DemoGaussianFactorGraph {
  var graph: FactorGraph
  var inputZero: PackedStorage
}

extension DemoGaussianFactorGraph: GaussianFactor {
  func errorVector(_ x: PackedStorage) -> PackedStorage {
    return graph.factors.errorVectors(at: x)
  }
  func applyLinearForward(_ x: PackedStorage) -> PackedStorage {
    return graph.factors.applyLinearForward(x)
  }
  func applyLinearTranspose(_ y: PackedStorage) -> PackedStorage {
    var x = inputZero
    graph.factors.applyLinearTranspose(y, into: &x)
    return x
  }
}

// MARK: - "Demo"

public typealias Factors = PackedStorage
public typealias VariableAssignments = PackedStorage
public typealias ErrorVectors = PackedStorage

public func runSimplePose2SLAM() {

// # Section 1: Set up the initial guess.

  var values = VariableAssignments()
  let var0: SimpleTypedID<Pose2> = values.storeDifferentiable(Pose2(0.5, 0.0, 0.2))
  let var1: SimpleTypedID<Pose2> = values.storeDifferentiable(Pose2(2.3, 0.1, -0.2))
  let var2: SimpleTypedID<Pose2> = values.storeDifferentiable(Pose2(4.1, 0.1, .pi / 2))
  let var3: SimpleTypedID<Pose2> = values.storeDifferentiable(Pose2(4.0, 2.0, .pi))
  let var4: SimpleTypedID<Pose2> = values.storeDifferentiable(Pose2(2.1, 2.1, -.pi / 2))
  let var5: SimpleTypedID<Int> = values.store(42)

  // Under the hood, `values` is a dictionary of homogeneous arrays:
  // [
  //   "Pose2": [Pose2(0.5, 0.0, 0.2), Pose2(2.3, 0.1, -0.2), ...]
  //   "Int"  : [42]
  // ]
  //
  // This allows fast access (just pointer arithmetic into the array buffer) if you know
  // statically what type you want.
  //
  // This also means that you only need one heap allocation per type of value.

  let initialGuess1: Pose2 = values[var0]
  // => Pose2(0.5, 0.0, 0.2)

  let initialGuess2: Int = values[var5]
  // => 42

// # Section 2: Set up the factor graph.

  var graph = FactorGraph()
  graph += DemoPriorFactor(var1, Pose2(0, 0, 0))
  graph += DemoBetweenFactor(var1, var0, Pose2(2.0, 0.0, .pi / 2))
  graph += DemoBetweenFactor(var2, var1, Pose2(2.0, 0.0, .pi / 2))
  graph += DemoBetweenFactor(var3, var2, Pose2(2.0, 0.0, .pi / 2))
  graph += DemoBetweenFactor(var4, var3, Pose2(2.0, 0.0, .pi / 2))

  // Under the hood, `graph` is a dictionary of homogeneous arrays, just like `values`.

// # Section 3: Linearize the graph and run CGLS.

  let linearApproximation: DemoGaussianFactorGraph = graph.linearized(at: values)
  let optimizer = CGLS(precision: 1e-6, max_iteration: 500)
  var dx = values.zeroTangentVector
  optimizer.optimize(linearApproximation, initial: &dx)

  print(graph.error(at: values))
  values.move(along: dx)
  print(graph.error(at: values))

// # Section 4: Performance.

  // Artificial benchmark:
  //   Intel Pose2SLAM dataset.
  //   Nonlinear solver: 10 iterations of Gauss-Newton.
  //   Linear solver: 500 iterations of CGLS, "DummyPreconditioner".
  //   Always run max iterations, to guarantee that each implementation does similar amounts of work.

  // Results:
  //   SwiftFusion `NonlinearFactorGraph`                110    seconds
  //   GTSAM                                               4.1  seconds
  //   SwiftFusion `PackedStorage`                         2.1  seconds
  //   SwiftFusion `PackedStorage` + `@_specialize`        0.38 seconds

  // Final errors:
  //   SwiftFusion `NonlinearFactorGraph`                0.9873763132951586
  //   SwiftFusion `PackedStorage`                       0.9873763132951586
  //   SwiftFusion `PackedStorage` + `@_specialize`      0.9873763132951586
  //   GTSAM                                             0.953261

  //

  // Configuraiton details:
  //   CPU: Intel(R) Xeon(R) CPU E5-1650 v4 @ 3.60GHz
  //   SwiftFusion:
  //     * "-O -cross-module-optimization"
  //   GTSAM:
  //     * "Release" mode
  //     * "-march=native"
  //     * "SLOW_BUT_CORRECT_EXPMAP"
  //     * modified to use unit noise
  //     * no TBB
  //     * no MKL

}

// # Section 5: Defining a factor.

struct DemoPriorFactor: DifferentiableFactor1 {

  // MARK: - Input variable ids.

  typealias InputIDs = VariableIDs1<Pose2>
  var inputId1: SimpleTypedID<Pose2>

  // MARK: - Factor data.

  let prior: Pose2

  // MARK: - Initializer.

  init(_ inputId1: SimpleTypedID<Pose2>, _ prior: Pose2) {
    self.inputId1 = inputId1
    self.prior = prior
  }

  // MARK: - Error function.

  typealias ErrorVector = Pose2.TangentVector

  @differentiable(wrt: value)
  func errorVector(at value: Pose2) -> Pose2.TangentVector {
    return noise(prior.localCoordinate(value))
  }

  // MARK: - Linearization.

  typealias LinearizedFactor = JacobianFactor1<Matrix3x3>

  func linearized(at value: Pose2) -> JacobianFactor1<Matrix3x3> {
    return JacobianFactor1(
      linearizing: self.errorVector,
      at: value,
      inputId1: TypedID(inputId1.perTypeID),
      makeJacobian: makeJacobian
    )
  }

}

struct FactorGraph {
  var factors: Factors = Factors()

  static func += <T: AnyDifferentiableFactor>(_ graph: inout Self, _ factor: T) {
    _ = graph.factors.storeDifferentiableFactor(factor)
  }

  mutating func append<T: AnyFactor>(discrete factor: T) {
    _ = factors.storeFactor(factor)
  }

  func error(at values: VariableAssignments) -> Double {
    return factors.error(at: values)
  }

  func linearized(at values: VariableAssignments) -> DemoGaussianFactorGraph {
    var linearized = FactorGraph()
    factors.linearize(at: values, into: &linearized)
    return DemoGaussianFactorGraph(graph: linearized, inputZero: values.zeroTangentVector)
  }
}

public func runGenericFactorGraph() {
  var values = PackedStorage()
  let variable1 = values.storeDifferentiable(Pose2())
  //let variable2 = values.storeDifferentiable(Pose2())

  var graph = FactorGraph()
  graph += DemoPriorFactor(variable1, Pose2(1, 0, 0))
  //graph += ((variable1, variable2), DemoBetweenFactor(Pose2(1, 0, 1)))

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
  graph += DemoPriorFactor(intelPoseSLAMFactorGraph.variableIDs[0]!, Pose2(0, 0, 0))
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
    graph += DemoBetweenFactor(variableIDs[frameIndex]!, variableIDs[measuredIndex]!, pose)
  }
}

// MARK: - Bee tracking.

struct DiscreteTransitionFactor: Factor2 {
  typealias InputIDs = VariableIDs2<Int, Int>
  var inputId1: SimpleTypedID<Int>
  var inputId2: SimpleTypedID<Int>

  /// The number of states.
  let stateCount: Int

  /// Entry `i * stateCount + j` is the probability of transitioning from state `j` to state `i`.
  let transitionMatrix: [Double]

  init(
    _ inputId1: SimpleTypedID<Int>,
    _ inputId2: SimpleTypedID<Int>,
    _ stateCount: Int,
    _ transitionMatrix: [Double]
  ) {
    precondition(transitionMatrix.count == stateCount * stateCount)
    self.inputId1 = inputId1
    self.inputId2 = inputId2
    self.stateCount = stateCount
    self.transitionMatrix = transitionMatrix
  }

  func error(at value1: Int, _ value2: Int) -> Double {
    return -log(transitionMatrix[value2 * stateCount + value1])
  }
}

struct SwitchingBetweenFactor: DifferentiableFactor3 {
  typealias InputIDs = VariableIDs3<Int, Pose2, Pose2>
  var inputId1: SimpleTypedID<Int>
  var inputId2: SimpleTypedID<Pose2>
  var inputId3: SimpleTypedID<Pose2>

  /// The number of states.
  let stateCount: Int

  /// The movement for each state.
  let movements: [Pose2]

  init(
    _ inputId1: SimpleTypedID<Int>,
    _ inputId2: SimpleTypedID<Pose2>,
    _ inputId3: SimpleTypedID<Pose2>,
    _ stateCount: Int,
    _ movements: [Pose2]
  ) {
    precondition(movements.count == stateCount)
    self.inputId1 = inputId1
    self.inputId2 = inputId2
    self.inputId3 = inputId3
    self.stateCount = stateCount
    self.movements = movements
  }

  typealias ErrorVector = Vector3

  @differentiable(wrt: (value2, value3))
  func errorVector(at value1: Int, _ value2: Pose2, _ value3: Pose2) -> Vector3 {
    let actualDifference = between(value2, value3)
    return noise(movements[value1].localCoordinate(actualDifference))
  }

  typealias LinearizedFactor = JacobianFactor2<Matrix3x3, Matrix3x3>

  func linearized(at value1: Int, _ value2: Pose2, _ value3: Pose2) -> JacobianFactor2<Matrix3x3, Matrix3x3> {
    return JacobianFactor2(
      linearizing: { self.errorVector(at: value1, $0, $1) },
      at: value2, value3,
      inputId1: TypedID(inputId2.perTypeID),
      inputId2: TypedID(inputId3.perTypeID),
      makeJacobian: makeJacobian
    )
  }
}

func noise(_ v: Vector3) -> Vector3 {
  let n = Double(5)
  return Vector3(n * v.x, n * v.y, n * v.z)
}

public func runBeeTrackingExample() {
  // Model parameters.
  let labelCount = 3
  let transitionMatrix: [Double] = [
    0.8, 0.1, 0.1,
    0.1, 0.8, 0.1,
    0.1, 0.1, 0.8
  ]
  let movements = [
    Pose2(1, 0, 0),       // go forwards
    Pose2(1, 0, .pi / 4), // turn left
    Pose2(1, 0, -.pi / 4)  // turn right
  ]

  // Generate some synthetic data.
  let actualLabels = generateSyntheticLabels(
    labelCount: labelCount, transitionMatrix: transitionMatrix, count: 30)
  //let actualLabels = Array(repeating: 0, count: 10)
  //    + Array(repeating: 1, count: 10)
  //    + Array(repeating: 2, count: 10)
  let observations = generateSyntheticObservations(
    labels: actualLabels,
    movements: movements,
    moveNoise: 0.0,
    observeNoise: 0.0
  )

  print(observations)

  // Create the initial guess: bee going straight for 30 steps.
  let initialGuessLabels = Array(repeating: 0, count: 30)
  let initialGuessPositions = Array(repeating: Pose2(0, 0, 0), count: 30)

  // Create variables, assigned to the initial guess.
  var values = VariableAssignments()
  let labelVars = initialGuessLabels.map { values.store($0) }
  let positionVars = initialGuessPositions.map { values.storeDifferentiable($0) }

  // Create the factor graph.
  var graph = FactorGraph()
  for i in 0..<(labelVars.count - 1) {
    graph.append(discrete: DiscreteTransitionFactor(
      labelVars[i], labelVars[i + 1], labelCount, transitionMatrix
    ))
    graph += SwitchingBetweenFactor(
      labelVars[i], positionVars[i], positionVars[i + 1], labelCount, movements
    )
  }
  for i in 0..<positionVars.count {
    graph += DemoPriorFactor(positionVars[i], observations[i])
  }

  func labels(_ values: VariableAssignments) -> [Int] {
    return labelVars.map { values[$0] }
  }

  // Run Metropolis-Hastings with a proposal that changes the labels and Pose2SLAMs the positions.
  let metropolisStepCount = 1000000
  var acceptCount = 0
  var currentError = graph.error(at: values)
  for metropolisStep in 0..<metropolisStepCount {
    if metropolisStep % 1000 == 0 {
      print("Actual labels : \(actualLabels)")
      print("Current labels: \(labels(values))")
      print("Current error: \(currentError)")
      print("Accept count: \(acceptCount) / \(metropolisStep)")
      print("")
    }

    var proposedValues = values

    // Randomly change one label.
    proposedValues[labelVars[Int.random(in: 0..<labelVars.count)]] = Int.random(in: 0..<labelCount)

    // Pose2SLAM to find new proposed positions.

    // Hack: Make the problem better conditioned by resetting all the positions to 0.
    for i in positionVars.indices {
      proposedValues[positionVars[i]] = Pose2(0, 0, 0)
    }
    gaussNewton(graph, &proposedValues)

    // Decide whether to accept or reject.
    let proposedError = graph.error(at: proposedValues)
    let logA = currentError - proposedError
    //print(logA)
    guard logA >= 0 || Double.random(in: 0...1) < exp(logA) else {
      // Proposal rejected.
      continue
    }

    // Proposal accepted.
    acceptCount += 1
    currentError = proposedError
    values = proposedValues
  }
}

func gaussNewton(_ graph: FactorGraph, _ values: inout VariableAssignments) {

  var currentError = graph.error(at: values)

  for _ in 0..<10 {
    let gfg = graph.linearized(at: values)
    let optimizer = CGLS(precision: 1e-6, max_iteration: 500)
    var dx = gfg.inputZero
    optimizer.optimize(gfg, initial: &dx)

    var newValues = values
    newValues.move(along: dx)
    let newError = graph.error(at: newValues)
    if newError > currentError - 1e-2 {
      return
    }

    currentError = newError
    values = newValues
  }
}

func randomRange(in range: Range<Int>) -> Range<Int> {
  let a = Int.random(in: range.startIndex..<(range.endIndex + 1))
  let b = Int.random(in: range.startIndex..<(range.endIndex + 1))
  return min(a, b)..<max(a, b)
}

func generateSyntheticLabels(
    labelCount: Int, transitionMatrix: [Double], count: Int) -> [Int] {
  var currentLabel = 0
  var labels: [Int] = []
  for _ in 0..<count {
    let r = Double.random(in: 0..<1)
    var newLabel = 0
    var cumSum = Double(0)
    for candidateNewLabel in 0..<labelCount {
      cumSum += transitionMatrix[candidateNewLabel * labelCount + currentLabel]
      if r < cumSum {
        newLabel = candidateNewLabel
        break
      }
    }
    currentLabel = newLabel
    labels.append(currentLabel)
  }
  return labels
 }


func generateSyntheticObservations(labels: [Int], movements: [Pose2], moveNoise: Double, observeNoise: Double) -> [Pose2] {
  var currentPosition = Pose2(0, 0, 0)
  var observations: [Pose2] = [currentPosition]
  observations.reserveCapacity(labels.count + 1)
  for label in labels {
    let moveNoisePose = moveNoise > 1e-2 ? Pose2(randomWithCovariance: moveNoise * eye(rowCount: 3)) : Pose2()
    let observeNoisePose = observeNoise > 1e-2 ? Pose2(randomWithCovariance: observeNoise * eye(rowCount: 3)) : Pose2()
    currentPosition = moveNoisePose * movements[label] * currentPosition
    observations.append(observeNoisePose * currentPosition)
  }
  return observations
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
