// swift-tools-version:5.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "SwiftFusion",
  platforms: [.macOS(.v10_14)],
  products: [
    // Products define the executables and libraries produced by a package, and make them visible to other packages.
    .library(
      name: "SwiftFusion",
      targets: ["SwiftFusion"]),
    .library(
      name: "BeeDataset",
      targets: ["BeeDataset"]),
    .executable(
      name: "Pose3SLAMG2O",
      targets: ["Pose3SLAMG2O"])
  ],
  dependencies: [
    // Dependencies declare other packages that this package depends on.
    // .package(url: /* package url */, from: "1.0.0"),
    .package(url: "https://github.com/google/swift-benchmark.git", .branch("master")),

    .package(url: "https://github.com/saeta/penguin.git", .branch("master")),

    .package(url: "https://github.com/ProfFan/tensorboardx-s4tf.git", from: "0.1.3"),
    .package(url: "https://github.com/apple/swift-tools-support-core.git", .branch("swift-5.2-branch")),
    .package(url: "https://github.com/tensorflow/swift-models.git", .branch("master")),
  ],
  targets: [
    // Targets are the basic building blocks of a package. A target can define a module or a test suite.
    // Targets can depend on other targets in this package, and on products in packages which this package depends on.
    .target(
      name: "SwiftFusion",
      dependencies: ["PenguinStructures"]),
    .target(
      name: "SwiftFusionBenchmarks",
      dependencies: [
        "Benchmark",
        "SwiftFusion",
      ]),
    .target(
      name: "BeeDataset",
      dependencies: [
        "SwiftFusion",
        .product(name: "Datasets", package: "swift-models"),
        .product(name: "ModelSupport", package: "swift-models"),
      ]),
    .target(
      name: "Pose3SLAMG2O",
      dependencies: ["SwiftFusion", "TensorBoardX", "SwiftToolsSupport"],
      path: "Examples/Pose3SLAMG2O"),
    .testTarget(
      name: "SwiftFusionTests",
      dependencies: ["SwiftFusion"]),
    .testTarget(
      name: "BeeDatasetTests",
      dependencies: ["BeeDataset"]),
  ])
