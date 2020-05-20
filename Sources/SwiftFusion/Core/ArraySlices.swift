// Copyright 2019 The SwiftFusion Authors. All Rights Reserved.
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

public extension Array where Element: Differentiable {
  struct Partitioned: Differentiable {
    var a: [Element]
    var b: [Element]
  }

  @differentiable(wrt: self)
  func differentiablePartition(_ index: Int) -> Partitioned {
    return Partitioned(a: Array(self[0..<index]), b: Array(self[index..<endIndex]))
  }

  @derivative(of: differentiablePartition)
  func vjpDifferentiablePartition(_ index: Int) -> (value: Partitioned, pullback: (Partitioned.TangentVector) -> TangentVector) {
    let partitioned = differentiablePartition(index)
    let aCount = partitioned.a.count
    let bCount = partitioned.b.count
    return (
      partitioned,
      {
        let aTv = $0.a.base.count > 0 ? $0.a.base : Array<Element.TangentVector>(repeating: .zero, count: aCount)
        let bTv = $0.b.base.count > 0 ? $0.b.base : Array<Element.TangentVector>(repeating: .zero, count: bCount)
        return TangentVector(aTv + bTv)
      }
    )
  }
}
