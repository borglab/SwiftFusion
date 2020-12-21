import _Differentiation
import SwiftFusion
import PenguinStructures

func reduced() {
  let sphere2500URL = try! cachedDataset("sphere2500.g2o")
  let sphere2500Dataset =  try! G2OReader.G2OFactorGraph(g2oFile3D: sphere2500URL)
  _ = ChordalInitialization.GetInitializations(
    graph: sphere2500Dataset.graph, ids: sphere2500Dataset.variableId)
}

reduced()

// Some attempts at reducing it that don't.

///// A factor that specifies a prior on a pose.
//public struct MyPriorFactor {
//  public let prior: Pose2
//
//  public init(_ id: TypedID<Pose2>, _ prior: Pose2) {
//    self.prior = prior
//  }
//
//  @differentiable
//  public func errorVector(_ x: Pose2) -> Pose2.TangentVector {
//    return .zero
//  }
//}
//
//func reduced2() {
//  let s2 = (0..<9).lazy.map { Double($0) }
//  let v2 = Matrix3(s2)
//  let f = RelaxedRotationFactorRot3(.init(0), .init(1), v2)
//  let (e, pb) = valueWithPullback(at: v2, v2, in: f.errorVector)
//  //print(f.errorVector(v2, v2))
//  print(e)
//  print(pb(Vector9(v2)))
//}
//
////reduced2()
//
//func reduced3() {
//  let v = Pose2(1, 2, 3)
//  let f = MyPriorFactor(.init(0), v)
//  let (e, pb) = valueWithPullback(at: v, in: f.errorVector)
//  print(e)
//  print(pb(Vector3(1, 1, 1)))
//}
//
////reduced3()
