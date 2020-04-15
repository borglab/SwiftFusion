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
import TensorFlow

/// A Factor Graph in its very general form.
///
/// Explanation
/// =============
/// A factor graph is a biparite graph that connects between *Factors* and *Values*, and
/// the way they are stored does not matter here.
///
public protocol FactorGraph {
  // TODO: find a better protocol
  associatedtype KeysType : Collection where KeysType.Element : SignedInteger
  associatedtype FactorsType : Collection where FactorsType.Element: Factor
  /// TODO(fan): Is this right?
  /// Or, do we need this at all? I think this would help to register keys to descriptions and
  /// help debugging and serialization, but I am not sure.
  var keys: KeysType { get }
  
  var factors: FactorsType { get }
}
