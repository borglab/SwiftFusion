import TensorFlow

public protocol TensorConvertible: Differentiable {
  @differentiable
  var tensor: Tensor<Double> {
    get
  }
}

// TODO(TF-1234): Remove this extension. It is a workaround.
extension TensorConvertible {
  /// Use this instead of "tensor" to worok around TF-1234.
  @differentiable var differentiableTensor: Tensor<Double> {
    return tensor
  }
}

extension Vector1: TensorConvertible {
  
}

extension Vector2: TensorConvertible {
  
}

extension Vector3: TensorConvertible {
  
}
