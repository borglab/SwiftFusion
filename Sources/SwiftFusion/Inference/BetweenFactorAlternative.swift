// Copyright 2020 The SwiftFusion Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import PenguinStructures

/// A BetweenFactor alternative that uses the Chordal (Frobenious) norm on rotation for Pose3
public struct BetweenFactorAlternative<JacobianRows: FixedSizeArray>:
  LinearizableFactor
  where JacobianRows.Element == Tuple2<Pose3.TangentVector, Pose3.TangentVector>
{
  public typealias Variables = Tuple2<Pose3, Pose3>

  public let edges: Variables.Indices
  public let difference: Pose3

  public init(_ startId: TypedID<Pose3, Int>, _ endId: TypedID<Pose3, Int>, _ difference: Pose3) {
    self.edges = Tuple2(startId, endId)
    self.difference = difference
  }

  public typealias ErrorVector = Vector12
  public func errorVector(_ start: Pose3, _ end: Pose3) -> ErrorVector {
    let actualMotion = between(start, end)
    let R = actualMotion.coordinate.rot.coordinate.R + (-1) * difference.rot.coordinate.R
    let t = actualMotion.t - difference.t
    
    // TODO(fan): Discuss with Marc why this would lead to failing CI
    // return Vector12(R[0, 0], R[0, 1], R[0, 2], R[1, 0], R[1, 1], R[1, 2], R[2, 0], R[2, 1], R[2, 2], t.x, t.y, t.z)
    return Vector12(R.s00, R.s01, R.s02, R.s10, R.s11, R.s12, R.s20, R.s21, R.s22, t.x, t.y, t.z)
  }

  // Note: All the remaining code in this factor is boilerplate that we can eventually eliminate
  // with sugar.
  
  public func error(at x: Variables) -> Double {
    return errorVector(at: x).squaredNorm
  }

  public func errorVector(at x: Variables) -> ErrorVector {
    return errorVector(x.head, x.tail.head)
  }

  public typealias Linearization = JacobianFactor<JacobianRows, ErrorVector>
  public func linearized(at x: Variables) -> Linearization {
    Linearization(linearizing: errorVector, at: x, edges: edges)
  }
}

public typealias Array8<T> = ArrayN<Array7<T>>
public typealias Array9<T> = ArrayN<Array8<T>>
public typealias Array10<T> = ArrayN<Array9<T>>
public typealias Array11<T> = ArrayN<Array10<T>>
public typealias Array12<T> = ArrayN<Array11<T>>

/// A Jacobian factor with 1 6-dimensional input and a 12-dimensional error vector.
public typealias JacobianFactor12x6_1 = JacobianFactor<Array12<Tuple1<Vector6>>, Vector12>

/// A Jacobian factor with 2 6-dimensional inputs and a 12-dimensional error vector.
public typealias JacobianFactor12x6_2 = JacobianFactor<Array12<Tuple2<Vector6, Vector6>>, Vector12>

/// A between factor on `Pose3`.
public typealias BetweenFactorAlternative3 = BetweenFactorAlternative<Array12<Tuple2<Vector6, Vector6>>>
