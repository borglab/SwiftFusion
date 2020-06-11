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

/// A heterogeneous array of values.
///
/// e.g. variable assignments, factor error vectors.
///
/// Note: This is just a temporary placeholder until we get the real heterogeneous array type. This
/// one is missing nice abstractions that let clients interact with it without knowing about
/// `contiguousStorage`.
struct VariableAssignments {
  /// Dictionary from variable type to contiguous storage for that type.
  var contiguousStorage: [ObjectIdentifier: AnyArrayBuffer<AnyArrayStorage>]
}

/// An identifier of a given abstract value with the value's type attached
///
/// - Parameter Value: the type of value this ID refers to.
/// - Parameter PerTypeID: a type that, given `Value`, identifies a given
///   logical value of that type.
///
/// Note: This is just a temporary placeholder until we get the real `TypedID` in penguin.
public struct TypedID<Value, PerTypeID: Equatable> {
  /// A specifier of which logical value of type `value` is being identified.
  let perTypeID: PerTypeID

  /// Creates an instance indicating the given logical value of type `Value`.
  init(_ perTypeID: PerTypeID) { self.perTypeID = perTypeID }
}
