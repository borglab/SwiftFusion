import TensorFlow
import XCTest
import PythonKit

import BeeTracking

class NNClassifierTests2: XCTestCase {
  /// Test that the hand-coded Jacobian for the decode method gives the same results as the
  /// AD-generated Jacobian.
  func testClassifier() {
    // Size of the images.
    let np = Python.import("numpy")
    let kHiddenDimension = 2
    let featureSize = 2
    // used to be 512
    print(softmax(Tensor([5,-5,10,-10])))

    let (imageHeight, imageWidth, imageChannels) =
      (8, 8, 1)
    var images: Tensor<Double> = .init(zeros: [6000, 8, 8, 1])
    images[3000...6000, 0..., 0...8, 0...1] = .init(ones: [3000,8,8,1])
    // print("image at index", images[3000,0...,0...,0...])
    var labels: Tensor<Int32> = .init(zeros: [6000])
    labels[3000...6000] = .init(ones: [3000])


    var classifier = NNClassifier(
      imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels,
      hiddenDimension: kHiddenDimension, latentDimension: featureSize
    )
    print("training data done")
    
    print("Training...")
    let rae: PretrainedNNClassifier = PretrainedNNClassifier(
      patches: images,
      labels: labels,
      given: PretrainedNNClassifier.HyperParameters(hiddenDimension: kHiddenDimension, latentDimension: featureSize, weightFile: "")
    )
    rae.save(to: "./classifier_weight_test_\(featureSize).npy")
    print("saved")



    

    
  


    //Tests: does it classify between 1 and 0.
    //Tests: does it classify an 8by8 white vs black images. feature size = 1 latent dim = 1. 
    //Tests: does it classify bees correctly.
    //Tracking factor: train classifier for a 3by3 image. 8by8.
    //Swift run 

    // Pass all the unit vectors throught the AD-generated pullback function and check that the
    // results match the hand-coded Jacobian.
    
  }
}
