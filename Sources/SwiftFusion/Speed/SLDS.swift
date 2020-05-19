#if false

// Based on some of [1].

/// In an SLDS, there is a hidden "position" that evolves over time, according to a set of n
/// linear movement models.
///
/// A sequence of hidden discrete variables determine which movement model is used at each time.
///
/// Let T be the number of time steps.
struct SLDSVariables {

  // MARK: - Definitely variables.

  /// movementModelIndex[t] is the index of the movement model used at time t.
  var movementModelIndex: [Int]

  /// position[t] is the position at time t.
  var position: [Vector]  // should be fixed-size vectors

  // MARK: - Maybe model parameters, maybe variables?

  /// This is an n x n matrix.
  /// entry (i, j) is the transition probability from movement model j to movement model i.
  var transitionMatrix: Matrix  // should be a fixed-size matrix

  /// movementMatrix[i] describes one time-step of movement under movement model i.
  /// e.g. position[t + 1] = movementMatrix[<the movement model index at time t>] * position[t].
  var movementMatrix: [Matrix]  // should be fixed-size matrices

  /// The observation at time t is `observationMatrix * position[t]`.
  var observationMatrix: Matrix  // should be a fixed-size matrix
}

struct SLDSParameters {
  /// observation[t] is the observation at time t.
  var observation: [Vector]  // should be a fixed-size vector
}

/// Describes T-1 "transition error" factors.
///
/// For t in [0, T-2], the t-th factor's inputs are
///   SLDSVariables.transitionMatrix : Matrix
///   SLDSVariables.movementModelIndex[t] : Int
///   SLDSVariables.movementModelIndex[t + 1] : Int
func transitionError(
  _ transitionMatrix: SLDSParameters,
  _ movementModelIndex1: Int,  // index of first movement model
  _ movementModelIndex2: Int  // index of next movement model
) -> Double {
  return -log(transitionMatrix[movementModelIndex2, movementModelIndex1])
}

/// Describes T-1 "movement error" factors.
///
/// For t in [0, T-2], the t-th factor's inputs are
///   SLDSVariables.movementMatrix : [Matrix]
///   SLDSVariables.movementModelIndex[t] : Int
///   SLDSVariables.position[t] : Vector
///   SLDSVariables.position[t + 1] : Vector
func movementError(
  _ movementMatrix: [Matrix],
  _ movementModelIndex: Int,
  _ position1: Vector,
  _ position2: Vector
) -> Double {
  return (position2 - movementMatrix[movementModelIndex] * position1).squaredNorm
}

/// Describs T "observation" factors.
///
/// For t in [0, T-1], the t-th factor's parameters are
///   SLDSParameters.observation[t] : Vector
///
/// For t in [0, T-1], the t-th factor's inputs are
///   SLDSVariables.position[t] : Vector
func observationError(
  _ observation: Vector,
  _ position: Vector
) -> Double {
  return (observation - parameters.observationMatrix * position).squaredNorm
}

// Now we might want to run Metropolis-Hastings sampling on this, as in [1].

// Operations that we need:
//
// Get a view of a graph where some variables are held constant.
//   - e.g. hold all nondifferentiable variables constant so that we can pass the graph to a solver
//     that uses differentiation.
//
// Belief Propagation
//   Factors send messages to adjacent variables.
//   Variables send messages to adjacent factors.
//   The type of the message is a probability distribution over the variable.
//     - Probability distributions over the same variable can be multiplied.
//     - Factors define "marginalization" functions that take all incoming messages and produce an
//       outgoing message along each edge.
//   In [1], all the probability distributions are normal distributions over vectors.
//

// [1] "Data-Driven MCMC for Learning and Inference in Switching Linear Dynamic Systems"
//      https://www.aaai.org/Papers/AAAI/2005/AAAI05-149.pdf

#endif
