import TensorFlow

public protocol TensorConvertible: Differentiable {
  @differentiable
  var tensor: Tensor<Double> {
    get
  }
}

extension Vector1: TensorConvertible {
  
}

extension Vector2: TensorConvertible {
  
}

extension Vector3: TensorConvertible {
  
}
