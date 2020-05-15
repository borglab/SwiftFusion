import SwiftFusion
import PenguinGraphs
import PenguinStructures
import TensorFlow
import XCTest

/// A factor graph tailored for a specific PoseSLAM problem.
struct PoseSLAMFactorGraph<Pose: LieGroup & UpdatedViewable>: FactorGraph {

  // MARK: - Graph strucutre.

  /// The data stored in a factor in this graph.
  enum FactorData: DefaultInitializable {
    case prior(prior: Pose)
    case between(difference: Pose)

    /// A case for the variable vertices, to work around the fact that there is currently no
    /// bipartite graph data structure that stores different data on each side.
    case none

    init() {
      self = .none
    }
  }

  /// The graph itself.
  var graph: AdjacencyList<FactorData, Empty, Int32>

  /// The number of variables in the graph.
  var variableCount: Int

  init(variableCount: Int) {
    self.graph = AdjacencyList()
    self.variableCount = variableCount
    for variable in 0..<variableCount {
      // TODO: We assume, without guarantee from the API, that vertex ids are consecutive integers
      // starting from 0. We should get a guarantee from the API or do something else.

      // Add a vertex representing the variable.
      let vertex = graph.addVertex(FactorData.none)

      // Assert that our assumption about the API hasn't been violated.
      assert(variable == variableIndex(of: vertex))
    }
  }

  private mutating func addUndirectedEdge(_ factor: Graph.VertexId, _ variable: Variables.Index) {
    _ = graph.addEdge(from: factor, to: vertexId(of: variable))
    _ = graph.addEdge(from: vertexId(of: variable), to: factor)
  }

  mutating func add(_ variable: Variables.Index, prior: Pose) {
    let factorVertexId = graph.addVertex(FactorData.prior(prior: prior))
    addUndirectedEdge(factorVertexId, variable)
  }

  mutating func add(
    _ variable1: Variables.Index,
    _ variable2: Variables.Index,
    between difference: Pose
  ) {
    let factorVertexId = graph.addVertex(FactorData.between(difference: difference))
    addUndirectedEdge(factorVertexId, variable1)
    addUndirectedEdge(factorVertexId, variable2)
  }

  // MARK: - Variables.

  /// A collection of values of the variables in the factor graph.
  struct Variables: IndexedVariables, UpdatedViewable {
    /// Values for all the variables.
    var poses: [Pose]

    /// Creates `Variables` from the given `poses`.
    init(_ poses: [Pose]) {
      self.poses = poses
    }

    /// Returns the value at `index`.
    subscript(index: Int) -> Pose {
      return poses[index]
    }

    /// The indices of all the variables.
    var indices: Array<Pose>.Indices {
      return poses.indices
    }

    /// An update to the variables.
    struct Update: IndexedVariables {
      let updates: [(Int, Pose.Update)]
      var indices: LazyMapCollection<[(Int, Pose.Update)], Int> {
        return updates.lazy.map { $0.0 }
      }
    }

    func updatedView(_ update: Update) -> Variables {
      var copy = self
      for (updatedIndex, updateValue) in update.updates {
        copy.poses[updatedIndex] = self.poses[updatedIndex].updatedView(updateValue)
      }
      return copy
    }
  }

  var variablesZero: Variables {
    return Variables(Array(repeating: Pose(), count: variableCount))
  }

  func vertexId(of variable: Variables.Index) -> Graph.VertexId {
    return Int32(variable)
  }

  func variableIndex(of vertex: Graph.VertexId) -> Variables.Index {
    return Int(vertex)
  }

  // MARK: - Factor error function.

  func error(of vertexId: Graph.VertexId, at point: Variables) -> Double {
    let variableIndices =
      graph.edges(from: vertexId).map { variableIndex(of: graph.destination(of: $0)) }
    switch graph[vertex: vertexId] {
    case .prior(prior: let prior):
      assert(variableIndices.count == 1)
      return prior.localCoordinate(point[variableIndices[0]]).squaredNorm
    case .between(difference: let difference):
      assert(variableIndices.count == 2)
      let actualDifference = between(point[variableIndices[1]], point[variableIndices[0]])
      return difference.localCoordinate(actualDifference).squaredNorm
    case .none:
      preconditionFailure("not a factor vertex")
    }
  }
}

extension Pose2: UpdatedViewable {
  public typealias Update = Self
  public func updatedView(_ update: Self) -> Self {
    return update
  }
}

/// An example proposal to use in the test.
extension PoseSLAMFactorGraph where Pose == Pose2 {
  /// Selects a variable uniformly at random, and some of the "nearby" variables, and applies the
  /// same random movement to all of them.
  func exampleProposal<Generator: RandomNumberGenerator>(
    startState: Variables,
    using generator: inout Generator
  ) -> (update: Variables.Update, logQRatio: Double) {
    // The movement.
    let movement = Pose2(randomWithCovariance: eye(rowCount: 3)) // TODO: this randomness not from the generator!

    // The first randomly selected variable.
    let selectedVariable = startState.indices.randomElement()!

    // Compute which ones are nearby.
    let affectedFactors = neighborFactors(of: selectedVariable)
    let nearbyVariables = affectedFactors.flatMap { neighborVariables(of: $0) }

    // Return an update that moves some of the nearby variables.
    var updates: [(Variables.Index, Pose)] = []
    updates.reserveCapacity(nearbyVariables.count)
    for nearbyVariable in nearbyVariables {
      guard nearbyVariable == selectedVariable || Double.random(in: 0...1) < 0.5 else { continue }
      updates.append((selectedVariable, movement * startState[nearbyVariable]))
    }
    return (update: Variables.Update(updates: updates), logQRatio: 0)
  }
}

final class MetropolisTests: XCTestCase {
  /// Tests that the Metropolis algorithm on a graph with a single varible and an single Gaussian
  /// prior produces a Gaussian distribution.
  func testSampleFromGaussian() {
    // Seeded RNG for reproducible results.
    var generator = ThreefryRandomNumberGenerator(seed: [42])

    // Construct a graph,  and initial values for variables.
    var graph = PoseSLAMFactorGraph<Pose2>(variableCount: 1)
    graph.add(0, prior: Pose2(0, 0, 0))
    var values = graph.variablesZero

    // Get some Metropolis samples.
    let sampleCount = 5000
    var samples: [Pose2] = []
    samples.reserveCapacity(sampleCount)
    for _ in 0..<sampleCount {
      let maybeUpdate = graph.metropolisStep(
        startState: values,
        proposal: graph.exampleProposal,
        using: &generator
      )
      if let update = maybeUpdate {
        values = values.updatedView(update)
      }
      samples.append(values[0])
    }

    // Calculate sample mean and covariance, in tangent space, as a simple test that the
    // distribution is what we expect.
    func mean(_ a: [Double]) -> Double {
      return a.reduce(0, +) / Double(a.count)
    }
    func covariance(_ a: [Double], _ b: [Double]) -> Double {
      precondition(a.count == b.count)
      return mean(zip(a, b).map(*)) - mean(a) * mean(b)
    }
    let tangentSpaceSamples = samples.map { Pose2().localCoordinate($0) }
    let xs = tangentSpaceSamples.map { $0.x }
    let ys = tangentSpaceSamples.map { $0.y }
    let zs = tangentSpaceSamples.map { $0.z }
    XCTAssertEqual(mean(xs), 0, accuracy: 0.1)
    XCTAssertEqual(mean(ys), 0, accuracy: 0.1)
    XCTAssertEqual(mean(zs), 0, accuracy: 0.1)
    XCTAssertEqual(covariance(xs, xs), 0.5, accuracy: 0.1)
    XCTAssertEqual(covariance(ys, ys), 0.5, accuracy: 0.1)
    XCTAssertEqual(covariance(zs, zs), 0.5, accuracy: 0.1)
    XCTAssertEqual(covariance(xs, ys), 0.0, accuracy: 0.1)
    XCTAssertEqual(covariance(xs, ys), 0.0, accuracy: 0.1)
    XCTAssertEqual(covariance(ys, zs), 0.0, accuracy: 0.1)
  }

  /// Test the Metropolis algorithm on a simple Pose2SLAM problem.
  func testMetropolisPose2SLAM() {
    // Seeded RNG for reproducible results.
    var generator = ThreefryRandomNumberGenerator(seed: [42])

    // Construct a graph.
    var graph = PoseSLAMFactorGraph<Pose2>(variableCount: 5)
    graph.add(0, prior: Pose2(0, 0, 0))
    graph.add(1, 0, between: Pose2(2.0, 0.0, .pi / 2))
    graph.add(2, 1, between: Pose2(2.0, 0.0, .pi / 2))
    graph.add(3, 2, between: Pose2(2.0, 0.0, .pi / 2))
    graph.add(4, 3, between: Pose2(2.0, 0.0, .pi / 2))

    // Initial estimates for the poses.
    var values = graph.variablesZero

    // Mix it a bit.
    let sampleCount = 5000
    var acceptanceCount = 0
    for _ in 0..<sampleCount {
      let maybeUpdate = graph.metropolisStep(
        startState: values,
        proposal: graph.exampleProposal,
        using: &generator
      )
      if let update = maybeUpdate {
        values = values.updatedView(update)
        acceptanceCount += 1
      }
    }

    // Metropolis with this proposal is terrible for this problem, so we're not getting anywhere
    // near the solution. So just assert that there weren't 0 or 100% acceptances so that this
    // test tests that Metropolis did some nontrivial work at least.
    XCTAssertGreaterThan(acceptanceCount, 0)
    XCTAssertLessThan(acceptanceCount, sampleCount)
  }
}
