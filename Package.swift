// swift-tools-version:5.3
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
    .package(name: "Benchmark", url: "https://github.com/google/swift-benchmark.git", from: "0.1.0"),

    .package(name: "Penguin", url: "https://github.com/saeta/penguin.git", .branch("main")),

    .package(name: "TensorBoardX", url: "https://github.com/dan-zheng/tensorboardx-s4tf.git", .branch("tensorflow-as-swiftpm-dependency")),
    .package(url: "https://github.com/apple/swift-tools-support-core.git", .branch("swift-5.2-branch")),
    .package(url: "https://github.com/apple/swift-argument-parser.git", from: "0.3.0"),
    .package(name: "Plotly", url: "https://github.com/vojtamolda/Plotly.swift", from: "0.4.0"),
    .package(name: "TensorFlow", url: "https://github.com/tensorflow/swift-apis", .branch("main")),
  ],
  targets: [
    // Targets are the basic building blocks of a package. A target can define a module or a test suite.
    // Targets can depend on other targets in this package, and on products in packages which this package depends on.
    .target(
      name: "SwiftFusion",
      dependencies: [
        .product(name: "PenguinStructures", package: "Penguin"),
        .product(name: "PenguinTesting", package: "Penguin"),
        .product(name: "PenguinParallelWithFoundation", package: "Penguin"),
        .product(name: "TensorFlow", package: "TensorFlow"),
      ],
      exclude: [
        "Core/VectorN.swift.gyb",
        "Inference/FactorBoilerplate.swift.gyb"
      ]
      ),
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
        "ModelSupport",
      ]),
    .target(
      name: "BeeTracking",
      dependencies: [
        "BeeDataset",
        "SwiftFusion",
      ]),
    .target(
      name: "Pose3SLAMG2O",
      dependencies: ["SwiftFusion", "TensorBoardX", .product(name: "SwiftToolsSupport", package: "swift-tools-support-core")],
      path: "Examples/Pose3SLAMG2O"),
    .target(
      name: "BeeTrackingTool",
      dependencies: [
        "BeeDataset",
        "BeeTracking",
        .product(name: "PenguinParallelWithFoundation", package: "Penguin"),
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
      ],
      path: "Examples/BeeTrackingTool"),
    .target(
      name: "OISTVisualizationTool",
      dependencies: [
        "BeeDataset",
        "BeeTracking",
        .product(name: "PenguinParallelWithFoundation", package: "Penguin"),
        "SwiftFusion",
        "Plotly",
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
      ],
    path: "Examples/OISTVisualizationTool"),
    .target(
      name: "Scripts",
      dependencies: [
        "BeeDataset",
        "BeeTracking",
        .product(name: "PenguinParallelWithFoundation", package: "Penguin"),
        "SwiftFusion",
        "Plotly",
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
      ],
      path: "Scripts",
      exclude: ["README.md"]
    ),
    .testTarget(
      name: "SwiftFusionTests",
      dependencies: [
        "SwiftFusion",
        "ModelSupport",
        .product(name: "PenguinTesting", package: "Penguin"),
      ],
      exclude: [
        "Datasets",
        "Core/VectorNTests.swift.gyb",
        "Image"
      ]
    ),
    .testTarget(
      name: "BeeDatasetTests",
      dependencies: ["BeeDataset"],
      exclude: ["fakeDataset"]
    ),
    .testTarget(
      name: "BeeTrackingTests",
      dependencies: [
        "BeeTracking",
        .product(name: "PenguinTesting", package: "Penguin"),
        "ModelSupport",
      ]),
    .target(
      name: "ModelSupport",
      dependencies: ["TensorFlow", "STBImage"]),
    .target(
      name: "STBImage",
      exclude: 
        ["CMakeLists.txt"]),
  ])
