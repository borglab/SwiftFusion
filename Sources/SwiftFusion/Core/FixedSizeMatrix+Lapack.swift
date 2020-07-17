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

/// Adds LAPACK methods to `FixedSizeMatrix`.

import CLapacke

extension FixedSizeMatrix {
  public func svd() -> (s: Rows.Element, u: Self, v: Self) {
    precondition(Self.isSquare)
    var s = Rows.Element.zero
    var u = self
    var vt = Self.zero
    var superb = Self.zero
    let m = Int32(Self.shape[0])
    let n = Int32(Self.shape[1])
    s.withUnsafeMutableBufferPointer { bS in
      u.withUnsafeMutableBufferPointer { bU in
        vt.withUnsafeMutableBufferPointer { bVT in
          superb.withUnsafeMutableBufferPointer { bSuperb in
            _ = LAPACKE_dgesvd(
              /*matrix_layout=*/LAPACK_ROW_MAJOR,
              /*jobu=*/Int8(Character("O").asciiValue!),
              /*jobv=*/Int8(Character("A").asciiValue!),
              /*m=*/m,
              /*n=*/n,
              /*a=*/bU.baseAddress,
              /*lda=*/n,
              /*s=*/bS.baseAddress,
              /*u=*/bU.baseAddress, // Note: this is never referenced.
              /*ldu=*/m,
              /*vt=*/bVT.baseAddress,
              /*ldvt=*/n,
              /*superb=*/bSuperb.baseAddress
            )
          }
        }
      }
    }
    return (s, u, vt.transposed())
  }
}
