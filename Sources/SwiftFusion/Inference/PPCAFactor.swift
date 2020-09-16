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

/// A factor that matches a patch on a latent variable.
/// The decomposition used here is in:
/// M. E. Tipping and C. M. Bishop, Probabilistic Principal Component Analysis,
/// Journal of the Royal Statistical Society. Series B, Vol. 61, No. 3 (1999), pp. 611-622
public struct PPCAFactor: LinearizableFactor1 {
  /// Type for an image patch with a fixed dimension
  public typealias Patch = Tensor10x10
  public let edges: Variables.Indices

  /// Measured Patch (Template)
  public let measured: Patch

  /// Weight Matrix
  public var W: Tensor<Double>
  
  /// Bias
  public var mu: Patch

  /// Constructor
  /// id: TypedID of the latent vector
  /// measured: Measured Patch (Template)
  /// W and mu are calculated by PPCA
  public init(_ id: TypedID<Vector5>, measured: Patch, W: Tensor<Double>, mu: Patch) {
    self.edges = Tuple1(id)
    self.measured = measured
    self.W = W
    self.mu = mu
  }

  /// Tangent Vector Type
  public typealias V0 = Vector5

  /// Error Vector for PPCA Factor
  /// e = W*a + mu - measured
  @differentiable
  public func errorVector(_ a: Vector5) -> Patch.TangentVector {
    return (mu + Patch(matmul(W, a.flatTensor.expandingShape(at: 1)).squeezingShape(at: 2)) - measured)
  }
}

public typealias JacobianFactor100x5_1 = JacobianFactor<Array<Tuple1<Vector5>>, Tensor10x10>
