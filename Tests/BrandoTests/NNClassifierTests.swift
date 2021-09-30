import TensorFlow
import XCTest
import PythonKit
import BeeDataset


import BeeTracking

class NNClassifierTests: XCTestCase {
  

  func testClassifier8by8() {
    // Size of the images.

    let np = Python.import("numpy")
    let kHiddenDimension = 2
    let featureSize = 2

    let (imageHeight, imageWidth, imageChannels) =
      (8, 8, 1)

    var classifier = NNClassifier(
      imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels,
      hiddenDimension: kHiddenDimension, latentDimension: featureSize
    )
    var weightsFile: String?
    if let weightsFile = weightsFile {
      classifier.load(weights: np.load(weightsFile, allow_pickle: true))
    } else {
      classifier.load(weights: np.load("./classifier_weight_test_\(featureSize).npy", allow_pickle: true))
      print("conv1", classifier.encoder_conv1)
      print("enc1",classifier.encoder1)
      print("enc2",classifier.encoder2)
      print("enc3",classifier.encoder3)      
      print("loaded")
    }
    let outblack = classifier.classify(.init(zeros: [1, 8, 8, 1]))
    let outwhite = classifier.classify(.init(ones: [1, 8, 8, 1]))
    print("zero image", classifier.classify(.init(zeros: [1, 8, 8, 1])))
    print("ones image", classifier.classify(.init(ones: [1, 8, 8, 1])))
    XCTAssertGreaterThan(Double(outblack[0,0])!,Double(outblack[0,1])!)
    XCTAssertGreaterThan(Double(outwhite[0,1])!,Double(outwhite[0,0])!)

    // zero image [[ 3.477267060877685, -3.477267060877686]]
    // ones image [[-8.87336098700629, 6.378658421614489]]
  }
  // Unit tests should not do the hevay lifting
  func testClassifier() {
    let np = Python.import("numpy")
    let kHiddenDimension = 512
    let featureSize = 8
    let batchSize = 500

    let (imageHeight, imageWidth, imageChannels) =
      (40, 70, 1)

    var classifier = NNClassifier(
      imageHeight: imageHeight, imageWidth: imageWidth, imageChannels: imageChannels,
      hiddenDimension: kHiddenDimension, latentDimension: featureSize
    )
    var weightsFile: String?
    if let weightsFile = weightsFile {
      classifier.load(weights: np.load(weightsFile, allow_pickle: true))
    } else {
      classifier.load(weights: np.load("./classifier_weight_\(featureSize).npy", allow_pickle: true))
    }
    let dataDir = URL(fileURLWithPath: "./OIST_Data")
    let dataset = OISTBeeVideo(directory: dataDir, length: 100)!
    // print("tests here1")
    let fgBoxes = dataset.makeForegroundBoundingBoxes(patchSize: (40, 70), batchSize: batchSize)
    print("here 1.5")
    let bgBoxes = dataset.makeBackgroundBoundingBoxes(patchSize: (40, 70), batchSize: batchSize)
    print("tests here2")
    let fgpatches = Tensor<Double>(stacking: fgBoxes.map { $0.frame!.patch(at: $0.obb)})
    let bgpatches = Tensor<Double>(stacking: fgBoxes.map { $0.frame!.patch(at: $0.obb)})

    let outfg = classifier.classify(fgpatches)
    let outbg = classifier.classify(bgpatches)
    let shapefg = outfg.shape
    let shapebg = outbg.shape
    print("fg", outfg)
    print("bg", outbg)
    XCTAssertEqual(outfg.shape, outbg.shape)
    XCTAssertEqual(outbg.shape, [batchSize, 2])

    var fgsum0 = 0.0
    var fgsum1 = 0.0
    var bgsum0 = 0.0
    var bgsum1 = 0.0
    for i in 0...batchSize-1 {
      fgsum0 += Double(outfg[i,0])!
      fgsum1 += Double(outfg[i,1])!
      bgsum0 += Double(outbg[i,0])!
      bgsum1 += Double(outbg[i,1])!
    }
    // Make sure classifier is working better than 50%
    XCTAssertGreaterThan(fgsum1,fgsum0)
    XCTAssertGreaterThan(bgsum0,bgsum1)




  }



}
