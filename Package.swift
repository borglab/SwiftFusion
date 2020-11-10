// swift-tools-version:5.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "SwiftFusion",
  platforms: [.macOS(.v10_15)],
  products: [
    // Products define the executables and libraries produced by a package, and make them visible to other packages.
    .library(
      name: "SwiftFusion",
      targets: ["SwiftFusion"]),
    .library(
      name: "BeeDataset",
      targets: ["BeeDataset"]),
    .library(
      name: "BeeTracking",
      targets: ["BeeTracking"]),
    .executable(
      name: "Pose3SLAMG2O",
      targets: ["Pose3SLAMG2O"])
  ],
  dependencies: [
    // Dependencies declare other packages that this package depends on.
    // .package(url: /* package url */, from: "1.0.0"),
    .package(url: "https://github.com/google/swift-benchmark.git", .branch("f70bf472b00aeaa05e2374373568c2fe459c11c7")),

    .package(url: "https://github.com/saeta/penguin.git", .branch("master")),

    .package(url: "https://github.com/ProfFan/tensorboardx-s4tf.git", from: "0.1.3"),
    .package(url: "https://github.com/apple/swift-tools-support-core.git", .branch("swift-5.2-branch")),
    .package(url: "https://github.com/tensorflow/swift-models.git", .branch("c67c9fc024d811e4134f379205ce49dd530f593a")),
    .package(url: "https://github.com/apple/swift-argument-parser", from: "0.2.0"),
    .package(url: "https://github.com/vojtamolda/Plotly.swift", from: "0.4.0"),
  ],
  targets: [
    // Targets are the basic building blocks of a package. A target can define a module or a test suite.
    // Targets can depend on other targets in this package, and on products in packages which this package depends on.
    .target(
      name: "SwiftFusion",
      dependencies: ["PenguinStructures", "PenguinTesting", "PenguinParallelWithFoundation"]),
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
        "Plotly",
        .product(name: "Datasets", package: "swift-models"),
        .product(name: "ModelSupport", package: "swift-models"),
      ]),
    .target(
      name: "BeeTracking",
      dependencies: [
        "BeeDataset",
        "SwiftFusion",
      ]),
    .target(
      name: "Pose3SLAMG2O",
      dependencies: ["SwiftFusion", "TensorBoardX", "SwiftToolsSupport"],
      path: "Examples/Pose3SLAMG2O"),
    .target(
      name: "BeeTrackingTool",
      dependencies: [
        "BeeDataset",
        "BeeTracking",
        "PenguinParallelWithFoundation",
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
      ],
      path: "Examples/BeeTrackingTool"),
    .target(
      name: "OISTVisualizationTool",
      dependencies: [
        "BeeDataset",
        "BeeTracking",
        "PenguinParallelWithFoundation",
        "SwiftFusion",
        "Plotly",
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
      ],
    path: "Examples/OISTVisualizationTool"),
    .testTarget(
      name: "SwiftFusionTests",
      dependencies: [
        "SwiftFusion",
        "PenguinTesting",
        .product(name: "ModelSupport", package: "swift-models"),
      ]),
    .testTarget(
      name: "BeeDatasetTests",
      dependencies: ["BeeDataset"]),
    .testTarget(
      name: "BeeTrackingTests",
      dependencies: ["BeeTracking"]),
  ])
