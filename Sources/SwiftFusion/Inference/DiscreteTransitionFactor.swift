//
//  DiscreteTransitionFactor.swift
//  
//
//  Frank Dellaert and Marc Rasi
//  July 2020

import Foundation
import PenguinStructures

/// A factor on two discrete labels evaluation the transition probability
struct DiscreteTransitionFactor : Factor {
  typealias Variables = Tuple2<Int, Int>
  
  /// The IDs of the variables adjacent to this factor.
  public let edges: Variables.Indices
  
  /// The number of states.
  let stateCount: Int
  
  /// Entry `i * stateCount + j` is the probability of transitioning from state `j` to state `i`.
  let transitionMatrix: [Double]
  
  init(
    _ inputId1: TypedID<Int>,
    _ inputId2: TypedID<Int>,
    _ stateCount: Int,
    _ transitionMatrix: [Double]
  ) {
    precondition(transitionMatrix.count == stateCount * stateCount)
    self.edges = Tuple2(inputId1, inputId2)
    self.stateCount = stateCount
    self.transitionMatrix = transitionMatrix
  }
  
  func error(at q: Variables) -> Double {
    let (label1, label2) = (q.head, q.tail.head)
    return -log(transitionMatrix[label2 * stateCount + label1])
  }
}
