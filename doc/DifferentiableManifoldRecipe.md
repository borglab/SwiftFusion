# Differentiable Manifold Recipe

This recipe describes how to express a differentiable manifold in Swift. The
main payoffs of using this recipe are:

 * You can choose arbitrary global and local coordinate systems for your
   manifolds.
 * Swift's Automatic Differentiation works with the coordinate system that you
   specify. In particular, the tangent vectors that flow through the
   automatically-generated derivatives are exactly the tangent vectors
   corresponding to the local coordinate system you specify.

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
/// A coordinate in a manifold's global coordinate system.
public protocol ManifoldCoordinate: Differentiable {
  /// The global coordinate corresponding to `local` in the chart centered around `self`.
  @differentiable(wrt: local)
  func retract(_ local: LocalCoordinate) -> Self

  /// The local coordinate corresponding to `global` in the chart centered around `self`.
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

Create a `struct` with a single stored property `var coordinateStorage` of
your coordinate type. Add an initializer `init(coordinateStorage:)` that
sets the stored coordinate. Add this method to the struct:

```swift
public mutating func move(along direction: Coordinate.LocalCoordinate) {
  coordinateStorage = coordinateStorage.retract(direction)
}
```

Finally, conform the struct to `Manifold`. This automatically adds a `var
coordinate` property that lets you access the point's coordinates and a
`init(coordinate:)` initializer that lets you create a new point with the given
coordinates. You should always use these methods rather than the
`coordinateStorage` methods when writing functions that use the manifold.

For example:

```swift
struct Rot2: Manifold {
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
