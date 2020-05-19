#if false

// Based on some of "Data-Driven MCMC for Learning and Inference in
// Switching Linear Dynamic Systems"
// (https://www.aaai.org/Papers/AAAI/2005/AAAI05-149.pdf)

/// In an SLDS, there is a hidden "position" that evolves over time, according to a set of n
/// linear movement models.
///
/// A sequence of hidden discrete variables determine which movement model is used at each time.
///
/// Let T be the number of time steps.
struct SLDSVariables {
  /// movementModelIndex[t] is the index of the movement model used at time t.
  /// Note: Sometimes we want to change this to a distribution over the movement model indices.
  var movementModelIndex: [Int]

  /// position[t] is the position at time t.
  var position: [Vector]  // should be fixed-size vectors

  /// This is an n x n matrix.
  /// entry (i, j) is the transition probability from movement model j to movement model i.
  var transitionMatrix: Matrix  // should be a fixed-size matrix

  /// movementMatrix[i] describes one time-step of movement under movement model i.
  /// e.g. position[t + 1] = movementMatrix[<the movement model index at time t>] * position[t].
  var movementMatrix: [Matrix]  // should be fixed-size matrices

  /// The observation at time t is `observationMatrix * position[t]`.
  var observationMatrix: Matrix  // should be a fixed-size matrix

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

/// The "distribution" version of the above factor.
func movementError(
  _ movementMatrix: [Matrix],
  _ movementModelIndexDistribution: [Double],  // now it is a distribution! 
  _ position1: Vector,
  _ position2: Vector
) -> Double {
  // some sum over all i of
  //  movementError(movementMatrix, i, position1, position2) * movementModelIndexDistribution[i]
}

/// Describs T "observation" factors.
///
/// For t in [0, T-1], the t-th factor's inputs are
///   SLDSVariables.observation[t] : Vector
///   SLDSVariables.position[t] : Vector
func observationError(
  _ observation: Vector,
  _ position: Vector
) -> Double {
  return (observation - parameters.observationMatrix * position).squaredNorm
}

// Now we want to run Metropolis-Hastings sampling on this.

// Operations that we want:
//
// Hold everything constant except for "positions". This gives you factors of the form:
//     (position[t + 1] - M[t] * position[t]).squaredNorm
//     (O[t] * position[t] - b[t]).squaredNorm
// where M[t], O[t] are fixed matrices and b[t] is a fixed vector. Now run "RTS Smoothing" which
// is some algorithm designed to find the optimal positions specifically in this structure of problem.
// It needs to see the fixed matrices and vectors.
//
// Calculate the total error given an assignment of variables.
//
// Change the "movementModelIndex[t]" from Int to a distribution over Int. Hold everything fixed except for
// the matrices (transitionMatrix, observationMatrix, and movementMatrix). Run a general purpose optimizer to
// find the optimal values of these. It'll specifically need derivatives of errors with respect to the matrices.

#endif
