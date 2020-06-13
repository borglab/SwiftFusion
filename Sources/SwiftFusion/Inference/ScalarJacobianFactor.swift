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

/// A Gaussian factor that scales its input by a scalar.
public struct ScalarJacobianFactor<ErrorVector: EuclideanVectorN>: NewGaussianFactor {
  public typealias Variables = Tuple1<ErrorVector>

  public let edges: Variables.Indices
  public let scalar: Double

  public func errorVector(at x: Variables) -> ErrorVector {
    return scalar * x.head
  }

  public func error(at x: Tuple1<ErrorVector>) -> Double {
    return errorVector(at: x).squaredNorm
  }

  public func errorVector_linearComponent(_ x: Variables) -> ErrorVector {
    return errorVector(at: x)
  }

  public func errorVector_linearComponent_adjoint(_ y: ErrorVector) -> Variables {
    return Tuple1(scalar * y)
  }

  public typealias Linearization = Self
  public func linearized(at x: Tuple1<ErrorVector>) -> ScalarJacobianFactor<ErrorVector> {
    return self
  }
}
