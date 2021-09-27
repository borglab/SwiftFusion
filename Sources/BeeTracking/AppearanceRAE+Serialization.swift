import PythonKit
import TensorFlow

extension Dense where Scalar: NumpyScalarCompatible {
  /// Loads weights and bias from the numpy arrays in `weights`.
  ///
  /// `weights[0]` is the dense weight matrix and `weights[1]` is the bias.
  mutating func load(weights: PythonObject) {
    let weight = Tensor<Scalar>(numpy: weights[0])!
    let bias = Tensor<Scalar>(numpy: weights[1])!
    print(self.weight.shape)
    print(weight.shape)
    precondition(
      self.weight.shape == weight.shape,
      "expected weight matrix \(self.weight.shape) but got \(weight.shape)")
    precondition(
      self.bias.shape == bias.shape, "expected bias \(self.bias.shape) but got \(bias.shape)")
    self.weight = weight
    self.bias = bias
    print("loaded")
  }

  /// The weight and bias as numpy arrays.
  ///
  /// `numpyWeights[0]` is the dense weight matrix and `numpyWeights[1]` is the bias.
  var numpyWeights: PythonObject {
    Python.list([self.weight.makeNumpyArray(), self.bias.makeNumpyArray()])
  }
}

extension Conv2D where Scalar: NumpyScalarCompatible {
  /// Loads filter and bias from the numpy arrays in `weights`.
  ///
  /// `weights[0]` is the filter and `weights[1]` is the bias.
  mutating func load(weights: PythonObject) {
    let filter = Tensor<Scalar>(numpy: weights[0])!
    let bias = Tensor<Scalar>(numpy: weights[1])!
    precondition(
      self.filter.shape == filter.shape,
      "expected filter matrix \(self.filter.shape) but got \(filter.shape)")
    precondition(
      self.bias.shape == bias.shape, "expected bias \(self.bias.shape) but got \(bias.shape)")
    self.filter = filter
    self.bias = bias
  }

  /// The filter and bias as numpy arrays.
  ///
  /// `numpyWeights[0]` is the filter and `numpyWeights[1]` is the bias.
  var numpyWeights: PythonObject {
    Python.list([self.filter.makeNumpyArray(), self.bias.makeNumpyArray()])
  }
}

extension DenseRAE {
  /// Loads model weights from the numpy arrays in `weights`.
  public mutating func load(weights: PythonObject) {
    self.encoder_conv1.load(weights: weights[0..<2])
    self.encoder1.load(weights: weights[2..<4])
    self.encoder2.load(weights: weights[4..<6])
    self.decoder1.load(weights: weights[6..<8])
    self.decoder2.load(weights: weights[8..<10])
    self.decoder_conv1.load(weights: weights[10..<12])
  }

  /// The model weights as numpy arrays.
  public var numpyWeights: PythonObject {
    [
      self.encoder_conv1.numpyWeights,
      self.encoder1.numpyWeights,
      self.encoder2.numpyWeights,
      self.decoder1.numpyWeights,
      self.decoder2.numpyWeights,
      self.decoder_conv1.numpyWeights
    ].reduce([], +)
  }

}


extension NNClassifier {
  /// Loads model weights from the numpy arrays in `weights`.
  public mutating func load(weights: PythonObject) {
    self.encoder_conv1.load(weights: weights[0..<2])
    self.encoder1.load(weights: weights[2..<4])
    self.encoder2.load(weights: weights[4..<6])
    self.encoder3.load(weights: weights[6..<8])
  }

  /// The model weights as numpy arrays.
  public var numpyWeights: PythonObject {
    [
      self.encoder_conv1.numpyWeights,
      self.encoder1.numpyWeights,
      self.encoder2.numpyWeights,
      self.encoder3.numpyWeights
    ].reduce([], +)
  }
}


extension SmallerNNClassifier {
  /// Loads model weights from the numpy arrays in `weights`.
  public mutating func load(weights: PythonObject) {
    self.encoder_conv1.load(weights: weights[0..<2])
    self.encoder1.load(weights: weights[2..<4])
    self.encoder2.load(weights: weights[4..<6])
  }

  /// The model weights as numpy arrays.
  public var numpyWeights: PythonObject {
    [
      self.encoder_conv1.numpyWeights,
      self.encoder1.numpyWeights,
      self.encoder2.numpyWeights,
    ].reduce([], +)
  }
}



extension LargerNNClassifier {
  /// Loads model weights from the numpy arrays in `weights`.
  public mutating func load(weights: PythonObject) {
    self.encoder_conv1.load(weights: weights[0..<2])
    self.encoder1.load(weights: weights[2..<4])
    self.encoder2.load(weights: weights[4..<6])
    self.encoder3.load(weights: weights[6..<8])
    self.encoder4.load(weights: weights[8..<10])

  }

  /// The model weights as numpy arrays.
  public var numpyWeights: PythonObject {
    [
      self.encoder_conv1.numpyWeights,
      self.encoder1.numpyWeights,
      self.encoder2.numpyWeights,
      self.encoder3.numpyWeights,
      self.encoder4.numpyWeights
    ].reduce([], +)
  }
}
