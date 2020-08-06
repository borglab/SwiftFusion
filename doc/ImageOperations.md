# Image Operations

## Coordinate systems

We have three coordinate systems for images:

- `(i, j)` are *integer* coordinates referring to pixels, and correspond to row
  and column in a matrix. The origin is `(i=0, j=0)` in the top-left.
- `(u, v)` are *float* coordinates, `u` is horizontal (associated with `j`) and
  `v` is vertical (associated with `i`).
- `(x, y)` are *float* coordinates, like `(u, v)`, but are "calibrated": `(x=0.0,
  y=0.0)` is the optical axis.

Pixel `(i, j)` is the square bounded by the corners with coordinates
`(u=float(j), v=float(i))` and `(u=float(j)+1.0, v=float(i)+1.0)`.

For example, pixel `(3, 1)`  the square bounded by the corners with coordinates
`(u=1.0, v=3.0)` and `(u=2.0, v=4.0)`.

Image dimensions are measured in two ways:

- `(rows, cols)` are *integer* dimensions corresponding to the `(i, j)` coordinates.
- `(width, height)` are *float* dimensions cooresponding to the `(u, v)` and `(x,
  y)` coordinates, with `(width=float(cols), height=float(rows))`.

For example, `(rows=3, cols=5)` corresponds to `(width=5, height=3)`. In this
image, the bottom-right pixel is `(i=2, j=4)`, corresponding to the square
bounded by `(u=4.0, v=2.0)` and `(u=5.0, v=3.0)`.

