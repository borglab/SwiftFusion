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

import TensorFlow
import PenguinStructures

/// A factor that specifies a patch on a latent variable.
public struct PPCAFactor: LinearizableFactor1 {
  public typealias Patch = Tensor10x10
  public let edges: Variables.Indices
  public let measured: Patch
  public var W: Tensor<Double>
  public var mu: Patch

  public init(_ id: TypedID<Vector5>, measured: Patch, W: Tensor<Double>, mu: Patch) {
    self.edges = Tuple1(id)
    self.measured = measured
    self.W = W
    self.mu = mu
  }

  public typealias V0 = Vector5
  @differentiable
  public func errorVector(_ a: Vector5) -> Patch.TangentVector {
    return (mu + Patch(matmul(W, a.flatTensor.expandingShape(at: 1)).squeezingShape(at: 2)) - measured)
  }
}

public typealias JacobianFactor100x5_1 = JacobianFactor<Array100<Tuple1<Vector5>>, Tensor10x10>

public typealias Array13<T> = ArrayN<Array12<T>>
public typealias Array14<T> = ArrayN<Array13<T>>
public typealias Array15<T> = ArrayN<Array14<T>>
public typealias Array16<T> = ArrayN<Array15<T>>
public typealias Array17<T> = ArrayN<Array16<T>>
public typealias Array18<T> = ArrayN<Array17<T>>
public typealias Array19<T> = ArrayN<Array18<T>>
public typealias Array20<T> = ArrayN<Array19<T>>
public typealias Array21<T> = ArrayN<Array20<T>>
public typealias Array22<T> = ArrayN<Array21<T>>
public typealias Array23<T> = ArrayN<Array22<T>>
public typealias Array24<T> = ArrayN<Array23<T>>
public typealias Array25<T> = ArrayN<Array24<T>>
public typealias Array26<T> = ArrayN<Array25<T>>
public typealias Array27<T> = ArrayN<Array26<T>>
public typealias Array28<T> = ArrayN<Array27<T>>
public typealias Array29<T> = ArrayN<Array28<T>>
public typealias Array30<T> = ArrayN<Array29<T>>
public typealias Array31<T> = ArrayN<Array30<T>>
public typealias Array32<T> = ArrayN<Array31<T>>
public typealias Array33<T> = ArrayN<Array32<T>>
public typealias Array34<T> = ArrayN<Array33<T>>
public typealias Array35<T> = ArrayN<Array34<T>>
public typealias Array36<T> = ArrayN<Array35<T>>
public typealias Array37<T> = ArrayN<Array36<T>>
public typealias Array38<T> = ArrayN<Array37<T>>
public typealias Array39<T> = ArrayN<Array38<T>>
public typealias Array40<T> = ArrayN<Array39<T>>
public typealias Array41<T> = ArrayN<Array40<T>>
public typealias Array42<T> = ArrayN<Array41<T>>
public typealias Array43<T> = ArrayN<Array42<T>>
public typealias Array44<T> = ArrayN<Array43<T>>
public typealias Array45<T> = ArrayN<Array44<T>>
public typealias Array46<T> = ArrayN<Array45<T>>
public typealias Array47<T> = ArrayN<Array46<T>>
public typealias Array48<T> = ArrayN<Array47<T>>
public typealias Array49<T> = ArrayN<Array48<T>>
public typealias Array50<T> = ArrayN<Array49<T>>
public typealias Array51<T> = ArrayN<Array50<T>>
public typealias Array52<T> = ArrayN<Array51<T>>
public typealias Array53<T> = ArrayN<Array52<T>>
public typealias Array54<T> = ArrayN<Array53<T>>
public typealias Array55<T> = ArrayN<Array54<T>>
public typealias Array56<T> = ArrayN<Array55<T>>
public typealias Array57<T> = ArrayN<Array56<T>>
public typealias Array58<T> = ArrayN<Array57<T>>
public typealias Array59<T> = ArrayN<Array58<T>>
public typealias Array60<T> = ArrayN<Array59<T>>
public typealias Array61<T> = ArrayN<Array60<T>>
public typealias Array62<T> = ArrayN<Array61<T>>
public typealias Array63<T> = ArrayN<Array62<T>>
public typealias Array64<T> = ArrayN<Array63<T>>
public typealias Array65<T> = ArrayN<Array64<T>>
public typealias Array66<T> = ArrayN<Array65<T>>
public typealias Array67<T> = ArrayN<Array66<T>>
public typealias Array68<T> = ArrayN<Array67<T>>
public typealias Array69<T> = ArrayN<Array68<T>>
public typealias Array70<T> = ArrayN<Array69<T>>
public typealias Array71<T> = ArrayN<Array70<T>>
public typealias Array72<T> = ArrayN<Array71<T>>
public typealias Array73<T> = ArrayN<Array72<T>>
public typealias Array74<T> = ArrayN<Array73<T>>
public typealias Array75<T> = ArrayN<Array74<T>>
public typealias Array76<T> = ArrayN<Array75<T>>
public typealias Array77<T> = ArrayN<Array76<T>>
public typealias Array78<T> = ArrayN<Array77<T>>
public typealias Array79<T> = ArrayN<Array78<T>>
public typealias Array80<T> = ArrayN<Array79<T>>
public typealias Array81<T> = ArrayN<Array80<T>>
public typealias Array82<T> = ArrayN<Array81<T>>
public typealias Array83<T> = ArrayN<Array82<T>>
public typealias Array84<T> = ArrayN<Array83<T>>
public typealias Array85<T> = ArrayN<Array84<T>>
public typealias Array86<T> = ArrayN<Array85<T>>
public typealias Array87<T> = ArrayN<Array86<T>>
public typealias Array88<T> = ArrayN<Array87<T>>
public typealias Array89<T> = ArrayN<Array88<T>>
public typealias Array90<T> = ArrayN<Array89<T>>
public typealias Array91<T> = ArrayN<Array90<T>>
public typealias Array92<T> = ArrayN<Array91<T>>
public typealias Array93<T> = ArrayN<Array92<T>>
public typealias Array94<T> = ArrayN<Array93<T>>
public typealias Array95<T> = ArrayN<Array94<T>>
public typealias Array96<T> = ArrayN<Array95<T>>
public typealias Array97<T> = ArrayN<Array96<T>>
public typealias Array98<T> = ArrayN<Array97<T>>
public typealias Array99<T> = ArrayN<Array98<T>>
public typealias Array100<T> = ArrayN<Array99<T>>