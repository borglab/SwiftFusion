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

    // There are some incompatible changes in penguin master but I need some new features so I have
    // temporarily switched this to a branch with the features but not the incompatible changes.
    // TODO(https://github.com/borglab/SwiftFusion/pull/154): Change back to `.branch("master")`.
    .package(url: "https://github.com/saeta/penguin.git", .branch("marcrasi-collection-protocols")),

    .package(url: "https://github.com/ProfFan/tensorboardx-s4tf.git", from: "0.1.3"),
    .package(url: "https://github.com/apple/swift-tools-support-core.git", .branch("swift-5.2-branch")),
    .package(url: "https://github.com/ProfFan/swift-models.git", .branch("temp_bench")),
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
    .target(
      name: "PatchSpeed",
      dependencies: ["SwiftFusion"],
      path: "Examples/PatchSpeed"),
    .testTarget(
      name: "SwiftFusionTests",
      dependencies: ["SwiftFusion"]),
    .testTarget(
      name: "BeeDatasetTests",
      dependencies: ["BeeDataset"]),
  ])
