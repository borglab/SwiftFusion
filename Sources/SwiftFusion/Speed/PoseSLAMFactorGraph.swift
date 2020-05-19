// Methods that factors provide:
//
// `errorVector` is a vector of errors. (The scalar `error` is the squared norm of this vector.)
//
// "gaussian factors" (aka "linear factors") are special factors that provide extra methods:
//   - applyLinearMap: applies a linear map to the inputs
//   - applyLinearMapTranspose: applies the tranpose of the linear map
// For gaussian factors, `errorVector(x)` is `applyLinearMap(x)` plus a constant vector. (Note
// that this is `DecomposedAffineFunction`).

// Operations that we do:
//
// Given a graph and an assignment of variables, get the total error.
//
// Given a graph and an assignment of variables, produce a new graph where each factor has been
// replaced with the derivative of the factor's error vector. This is a "gaussian factor graph".
//
// Given a gaussian factor graph, apply its linear map, its transposed linear map, and get its
// current error vector at an assignment of variables. (This is `DecomposedAffineFunction` again).
//   - each of these operations just means applying all of the factors' corresponding operations
//     to the variables

// Other important things:
//
// In gaussian factor graphs, the variables are collections of flat vectors, e.g:
//   [
//     Vector3(1, 2, 3),
//     Vector4(4, 5, 6, 7)
//   ]
// The "gaussian factors" want to receive their inputs as the correct vector type (e.g. Vector3,
// Vector4).
// The current implementation makes this fast by flattening these out into a single array of
// Doubles.
// A "type erase at the top level" implementation might actually be faster because you don't have
// to store or look up as many offsets.

public struct Pose2SLAMFactorGraph
{
  public typealias Pose2Values = [Pose2]

  public var priors: PriorFactors = PriorFactors()
  public var priorInputs: [Pose2Values.Index] = []

  public var betweens: BetweenFactors = BetweenFactors()
  public var betweenInputs: [(Pose2Values.Index, Pose2Values.Index)] = []

  public init() {}

  public func error(at values: Pose2Values) -> Double {
    var result = Double(0)
    for i in priorInputs.indices {
      result += priors.errorVector(of: i, at: ArraySubsettable1(values)[priorInputs[i]]).squaredNorm
    }
    for i in betweens.indices {
      result += betweens.errorVector(of: i, at: ArraySubsettable2(values)[betweenInputs[i]]).squaredNorm
    }
    return result
  }

  public func linearized(at values: Pose2Values) -> Pose2SLAMMaterializedGaussianFactorGraph {
    var linear = Pose2SLAMMaterializedGaussianFactorGraph(
      priorCount: priors.count,
      betweenCount: betweens.count,
      inputCount: values.count
    )
    for i in priorInputs.indices {
      linear.priors.addFactor(
        linearizing: { priors.errorVector(of: i, at: $0) },
        at: ArraySubsettable1(values)[priorInputs[i]],
        inputIndices: priorInputs[i],
        outputIndices: i
      )
    }
    for i in betweenInputs.indices {
      linear.betweens.addFactor(
        linearizing: { betweens.errorVector(of: i, at: $0) },
        at: ArraySubsettable2(values)[betweenInputs[i]],
        inputIndices: betweenInputs[i],
        outputIndices: i
      )
    }
    return linear
  }
}

public protocol Subsettable {
  associatedtype SubsetIndices
  associatedtype Subset
  subscript(_ indices: SubsetIndices) -> Subset { get set }
}

public struct ArraySubsettable1<Element: Differentiable>: Subsettable {
  public var base: Array<Element>
  public init(_ base: Array<Element>) {
    self.base = base
  }
  public subscript(_ indices: Int) -> Element {
    get {
      return base[indices]
    }
    set(newValue) {
      base[indices] = newValue
    }
  }
}

public struct ArraySubsettable2<Element: Differentiable>: Subsettable {
  public var base: Array<Element>
  public init(_ base: Array<Element>) {
    self.base = base
  }
  public subscript(_ indices: (Int, Int)) -> ValuesSubset2<Element> {
    get {
      return ValuesSubset2(base[indices.0], base[indices.1])
    }
    set(newValue) {
      base[indices.0] = newValue.value1
      base[indices.1] = newValue.value2
    }
  }
}

public struct ValuesSubset2<Value> {
  var value1Storage: Value
  var value2Storage: Value

  public var value1: Value {
    @differentiable(where Value: Differentiable)
    get {
      return value1Storage
    }
    set(newValue) {
      value1Storage = newValue
    }
  }

  public var value2: Value {
    @differentiable(where Value: Differentiable)
    get {
      return value2Storage
    }
    set(newValue) {
      value2Storage = newValue
    }
  }

  @differentiable(where Value: Differentiable)
  public init(_ value1: Value, _ value2: Value) {
    self.value1Storage = value1
    self.value2Storage = value2
  }
}

extension ValuesSubset2: Equatable where Value: Equatable {}
extension ValuesSubset2: AdditiveArithmetic where Value: AdditiveArithmetic {}

extension ValuesSubset2: Differentiable where Value: Differentiable {
  public typealias TangentVector = ValuesSubset2<Value.TangentVector>

  public mutating func move(along direction: TangentVector) {
    value1Storage.move(along: direction.value1)
    value2Storage.move(along: direction.value2)
  }

  @derivative(of: value1)
  @usableFromInline
  func vjpValue1() -> (value: Value, pullback: (Value.TangentVector) -> TangentVector) {
    return (value1, { TangentVector($0, Value.TangentVector.zero) })
  }

  @derivative(of: value2)
  @usableFromInline
  func vjpValue2() -> (value: Value, pullback: (Value.TangentVector) -> TangentVector) {
    return (value2, { TangentVector(Value.TangentVector.zero, $0) })
  }

  @derivative(of: init)
  @usableFromInline
  static func vjpInit(_ value1: Value, _ value2: Value) ->
    (value: Self, pullback: (TangentVector) -> (Value.TangentVector, Value.TangentVector))
  {
    return (Self(value1, value2), { ($0.value1, $0.value2) })
  }
}

extension ValuesSubset2: FixedDimensionVector where Value: FixedDimensionVector {
  public static var dimension: Int { return 2 * Value.dimension }
  public subscript(_ index: Int) -> Double {
    get {
      if index < Value.dimension {
        return value1[index]
      }
      return value2[index - Value.dimension]
    }
    set(newValue) {
      if index < Value.dimension {
        value1[index] = newValue
        return
      }
      value2[index - Value.dimension] = newValue
    }
  }
  public static var zero: Self {
    return Self(Value.zero, Value.zero)
  }
  public func write(to array: inout Vector, at index: Int) {
    value1.write(to: &array, at: index)
    value2.write(to: &array, at: index + Value.dimension)
  }
}

public struct PriorFactors
{
  public var parameters: [Pose2] = []

  public init() {}

  public var count: Int { return parameters.count }
  public var indices: Range<Int> { return parameters.indices }

  @differentiable(wrt: input)
  public func errorVector(of factor: Int, at input: Pose2) -> Pose2.TangentVector {
    let prior = parameters[factor]
    return prior.localCoordinate(input)
  }
}

public struct BetweenFactors
{
  public var parameters: [Pose2] = []

  public init() {}

  public var count: Int { return parameters.count }
  public var indices: Range<Int> { return parameters.indices }

  @differentiable(wrt: input)
  public func errorVector(of factor: Int, at input: ValuesSubset2<Pose2>) -> Pose2.TangentVector {
    let actualDifference = between(input.value1, input.value2)
    let expectedDifference = parameters[factor]
    return expectedDifference.localCoordinate(actualDifference)
  }
}

public struct Pose2SLAMMaterializedGaussianFactorGraph
{
  public var priors: HomogeneousGaussianFactorGraph<
    HomogeneousVectorSubsettable1<Pose2.TangentVector>,
    HomogeneousVectorSubsettable1<Pose2.TangentVector>
  >
  public var betweens: HomogeneousGaussianFactorGraph<
    HomogeneousVectorSubsettable2<Pose2.TangentVector>,
    HomogeneousVectorSubsettable1<Pose2.TangentVector>
  >

  public init(priorCount: Int, betweenCount: Int, inputCount: Int) {
    self.priors = HomogeneousGaussianFactorGraph(factorCount: priorCount, inputCount: inputCount)
    self.betweens = HomogeneousGaussianFactorGraph(factorCount: betweenCount, inputCount: inputCount)
  }

  public var zeroInput: HomogeneousVectorSubsettable1<Pose2.TangentVector> { return self.priors.zeroInput }
}

public protocol ZeroInitializable {
  init(zeros dimension: Int)
}

public struct HomogeneousGaussianFactorGraph<
  Input: Subsettable & ZeroInitializable & EuclideanVectorSpace,
  Output: Subsettable & ZeroInitializable & EuclideanVectorSpace
> where Input.Subset: FixedDimensionVector, Output.Subset: FixedDimensionVector {
  public var inputs: [Input.SubsetIndices] = []
  public var outputs: [Output.SubsetIndices] = []
  public var matrixScalars: [Double] = []
  public var error: Output

  public let factorCount: Int
  public let inputCount: Int

  public var matrixScalarCountPerFactor: Int { return Input.Subset.dimension * Output.Subset.dimension }
  public var errorScalarCountPerFactor: Int { return Output.Subset.dimension }

  public var zeroInput: Input { return Input(zeros: inputCount) }
  public var zeroOutput: Output { return Output(zeros: factorCount) }

  public init(factorCount: Int, inputCount: Int) {
    self.factorCount = factorCount
    self.inputCount = inputCount
    self.error = Output(zeros: factorCount)  // TODO: Useless initialization to zero.
    inputs.reserveCapacity(factorCount)
    outputs.reserveCapacity(factorCount)
    matrixScalars.reserveCapacity(factorCount * matrixScalarCountPerFactor)
  }

  public mutating func addFactor<A>(
    linearizing f: @differentiable (A) -> Output.Subset,
    at input: A,
    inputIndices: Input.SubsetIndices,
    outputIndices: Output.SubsetIndices
  ) where A.TangentVector == Input.Subset, Output.Subset.TangentVector: FixedDimensionVector {
    precondition(inputs.count < factorCount)
    inputs.append(inputIndices)
    outputs.append(outputIndices)
    let (value, pb) = valueWithPullback(at: input, in: f)
    error[outputIndices] = value
    for basisVector in Output.Subset.TangentVector.standardBasis {
      let v = pb(basisVector)
      for i in 0..<Input.Subset.dimension {
        matrixScalars.append(v[i])
      }
    }
    assert(matrixScalars.count == inputs.count * matrixScalarCountPerFactor)
  }

  public func factorMatrix(_ factor: Int) -> ArraySlice<Double> {
    return matrixScalars[(factor * matrixScalarCountPerFactor)..<((factor + 1) * matrixScalarCountPerFactor)]
  }
}

extension HomogeneousGaussianFactorGraph: DecomposedAffineFunction {
  public func applyLinearForward(_ input: Input) -> Output {
    var output = Output(zeros: factorCount)
    for factor in inputs.indices {
      let inputSubset = input[inputs[factor]]
      var outputSubset = output[outputs[factor]]
      let matrix = factorMatrix(factor)
      var matrixIndex = matrix.startIndex
      for i in 0..<Output.Subset.dimension {
        for j in 0..<Input.Subset.dimension {
          outputSubset[i] += matrix[matrixIndex] * inputSubset[j]
          matrixIndex += 1
        }
      }
      output[outputs[factor]] = outputSubset
    }
    return output
  }

  public func applyLinearAdjoint(_ output: Output) -> Input {
    var input = Input(zeros: inputCount)
    for factor in inputs.indices {
      var inputSubset = input[inputs[factor]]
      let outputSubset = output[outputs[factor]]
      let matrix = factorMatrix(factor)
      var matrixIndex = matrix.startIndex
      for i in 0..<Output.Subset.dimension {
        for j in 0..<Input.Subset.dimension {
          inputSubset[j] += matrix[matrixIndex] * outputSubset[i]
          matrixIndex += 1
        }
      }
      input[inputs[factor]] = inputSubset
    }
    return input
  }

  public var bias: Output {
    return error
  }
}

extension Pose2SLAMMaterializedGaussianFactorGraph: DecomposedAffineFunction {
  /// The linear component of the affine function.
  public func applyLinearForward(_ x: HomogeneousVectorSubsettable1<Pose2.TangentVector>) -> Output {
    return Output(
      priorErrors: priors.applyLinearForward(x),
      betweenErrors: betweens.applyLinearForward(HomogeneousVectorSubsettable2(x.base))
    )
  }

  /// The linear adjoint of the linear component of the affine function.
  public func applyLinearAdjoint(_ y: Output) -> HomogeneousVectorSubsettable1<Pose2.TangentVector> {
    return HomogeneousVectorSubsettable1(
      priors.applyLinearAdjoint(y.priorErrors).base + betweens.applyLinearAdjoint(y.betweenErrors).base
    )
  }

  public var bias: Output {
    return Output(priorErrors: priors.error, betweenErrors: betweens.error)
  }

  public struct Output: EuclideanVectorSpace {
    public var priorErrors: HomogeneousVectorSubsettable1<Pose2.TangentVector>
    public var betweenErrors: HomogeneousVectorSubsettable1<Pose2.TangentVector>

    public var squaredNorm: Double {
      return priorErrors.squaredNorm + betweenErrors.squaredNorm
    }
  }
}

public struct HomogeneousVectorSubsettable1<Element: Differentiable & FixedDimensionVector>: Subsettable {
  public var base: Vector
  public init(_ base: Vector) {
    self.base = base
  }

  private func range(_ index: Int) -> Range<Int> {
    return (Element.dimension * index)..<(Element.dimension * (index + 1))
  }

  public subscript(_ indices: Int) -> Element {
    get {
      return Element(base.scalars[range(indices)])
    }
    set(newValue) {
      newValue.write(to: &base, at: Element.dimension * indices)
    }
  }
}

extension HomogeneousVectorSubsettable1: EuclideanVectorSpace {
  public var squaredNorm: Double { return base.squaredNorm }
}

extension HomogeneousVectorSubsettable1: ZeroInitializable {
  public init(zeros dimension: Int) {
    self.base = Vector(zeros: dimension * Element.dimension)
  }
}

public struct HomogeneousVectorSubsettable2<Element: Differentiable & FixedDimensionVector>: Subsettable {
  public var base: Vector
  public init(_ base: Vector) {
    self.base = base
  }

  private func range(_ index: Int) -> Range<Int> {
    return (Element.dimension * index)..<(Element.dimension * (index + 1))
  }

  public subscript(_ indices: (Int, Int)) -> ValuesSubset2<Element> {
    get {
      return ValuesSubset2(
        Element(base.scalars[range(indices.0)]),
        Element(base.scalars[range(indices.1)])
      )
    }
    set(newValue) {
      newValue.value1.write(to: &base, at: Element.dimension * indices.0)
      newValue.value2.write(to: &base, at: Element.dimension * indices.1)
    }
  }
}

extension HomogeneousVectorSubsettable2: EuclideanVectorSpace {
  public var squaredNorm: Double { return base.squaredNorm }
}

extension HomogeneousVectorSubsettable2: ZeroInitializable {
  public init(zeros dimension: Int) {
    self.base = Vector(zeros: dimension * Element.dimension)
  }
}
