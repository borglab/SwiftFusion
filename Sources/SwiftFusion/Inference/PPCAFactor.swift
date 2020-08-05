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

public typealias JacobianFactor100x5_1 = JacobianFactor<Array<Tuple1<Vector5>>, Tensor10x10>
