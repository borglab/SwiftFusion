import Quote

public protocol JacobianEvaluatable {
    associatedtype ValueType
    func jacobianAt(n: Int, m: Int) -> ValueType
}