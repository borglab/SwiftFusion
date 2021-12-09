
public protocol Optimizer {
  mutating func optimize(graph: FactorGraph, initial: inout VariableAssignments) -> ()
  
}