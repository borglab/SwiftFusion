# `init(_ source)`

The `init(_ source)` does not work with tuples of dynamically sized vectors.

But I think the only place where I need tuples to be vectors is when I'm forming
the rows of the jacobian and I do not need the `init(_ source)` there.

I wonder where I do need the `init(_ source)`. Maybe it is just for like
conversion and concatenation and that only happens on legacy fixed size vectors.
