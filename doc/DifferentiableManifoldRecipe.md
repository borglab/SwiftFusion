# Differentiable Manifold Recipe

This recipe describes how to express a differentiable manifold in Swift. The
main payoffs of using this recipe are:

 * You can choose arbitrary global and local coordinate systems for your
   manifolds.
 * Swift's Automatic Differentiation works with the coordinate system that you
   specify. In particular, the tangent vectors that flow through the
   automatically-generated derivatives are exactly the tangent vectors
   corresponding to the local coordinate system you specify.

Note: We want to make the recipe simpler. See
[issue #45](https://github.com/borglab/SwiftFusion/issues/45).

## Steps

### 1. Choose a global coordinate system.

Create a `struct` whose values represent points on the manifold. Every point on
the manifold must be representable by at least one `struct` value. Every
`struct` value must represent at most one point on the manifold.

For example, both of these `struct`s are acceptable global coordinate systems
for `SO(2)`, the manifold of 2D rotations:

```swift
struct SinCosGlobalCoordinate {
  /// The cosine of the angle of the rotation.
  var c: Float

  /// The sine of the angle of the rotation.
  var s: Float
}

struct AngleGlobalCoordinate {
  /// The angle of the rotation.
  var theta: Float
}
```

### 2. Choose the local coordinate type.

Choose an `n`-dimensional vector type to represent to represent the manifold's
local coordinates.

For example, `Float` and `Vector1` are both acceptable types for `SO(2)` local
coordinates.

### 3. Conform to the ManifoldCoordinate protocol.

The coordinate `struct` you chose in step 1 should conform to the
`ManifoldCoordinate` protocol, which has these requirements:

```swift
public protocol ManifoldCoordinate: Differentiable {
  /// The local coordinate type of the manifold.
  ///
  /// This is the `TangentVector` of the `Manifold` wrapper type.
  ///
  /// Note that this is not the same type as `Self.TangentVector`.
  associatedtype LocalCoordinate: VectorSpace

  /// Diffeomorphism between a neigborhood of `Localcoordinate.zero` and `Self`.
  ///
  /// Satisfies the following properties:
  /// - `retract(LocalCoordinate.zero) == self`
  /// - There exists an open set `B` around `LocalCoordinate.zero` such that
  ///   `localCoordinate(retract(b)) == b` for all `b \in B`.
  @differentiable(wrt: local)
  func retract(_ local: LocalCoordinate) -> Self

  /// Inverse of `retract`.
  ///
  /// Satisfies the following properties:
  /// - `localCoordinate(self) == LocalCoordinate.zero`
  /// - There exists an open set `B` around `self` such that `localCoordinate(retract(b)) == b` for all
  ///   `b \in B`.
  @differentiable(wrt: global)
  func localCoordinate(_ global: Self) -> LocalCoordinate
}
```

(See the code for more detailed requirements and documentation).

The implementations of `global` and `local` determine how the manifold's global
and local coordinates are related, and they therefore determine the behavior of
the manifold's tangent vectors during differentiation.

Note: These are also known as the "exponential map" and "logarithm" in
differential geometry.

### Step 4. Create the manifold type.

Create a struct with a stored property and initalizer as follows:
```
var coordinateStorage: ${your coordinate type}

init(coordinateStorage: ${your coordinate type} {
  self.coordinateStorage = coordinateStorage
}
```

Conform the struct to `Manifold` and specify two typealiases:
```
public typealias Coordinate = {your coordinate type}
public typealias TangentVector = {your local coordinate type}
```

The `Manifold` conformance makes your struct differentiable with tangent space
given by the local coordinate type that you specified.

The `Manifold` conformance also adds a computed property and initializer to your
struct:
```
// Automatically defined: don't define this yourself.
@differentiable
var coordinate: Coordinate

// Automatically defined: don't define this yourself.
@differentiable
init(coordinate: Coordinate)
```

You should always use these rather than the `coordinateStorage` methods when
writing functions on your manifold. If you use a `coordinateStorage` method,
you'll get an error telling you that it is not differentiable. (TODO: Actually,
it's an internal compiler error now: https://bugs.swift.org/browse/TF-969).

Finally, add this method to the struct:
```swift
public mutating func move(along direction: TangentVector) {
  coordinateStorage = coordinateStorage.retract(direction)
}
```
(TODO: It should be possible to make this a default implementation so that users
do not need to specify it.)

For example:

```swift
struct Rot2: Manifold {
  public typealias Coordinate = SinCosGlobalCoordinate
  public typealias TangentVector = Vector1

  var coordinateStorage: SinCosGlobalCoordinate

  init(coordinateStorage: SinCosGlobalCoordinate) {
    self.coordinateStorage = coordinateStorage
  }

  public mutating func move(along direction: Coordinate.LocalCoordinate) {
    coordinateStorage = coordinateStorage.retract(direction)
  }
}
```

### Step 5: Done.

Now you can write functions on the manifold and they automatically get
derivatives with tangent vectors corresponding to the local coordinates you
specified. For example:

```swift
extension Rot2 {
  @differentiable
  var theta: Float {
    // atan2 in the stdlib does not have the derivative implementation
    // so we need to use a wrapped version that does
    atan2wrap(coordinate.s, coordinate.c)
  }
}

gradient(at: Rot2(2)) { $0.theta }  // => 1
```

This works because all functions on the manifold access manifold coordinates
through the `coordinate` property, and the derivatives of `coordinate`
(specified in `Manifold.swift`) take the specified `global` and `local`
functions into account.
