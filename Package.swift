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
    .executable(
      name: "Pose2SLAMG2O",
      targets: ["Pose2SLAMG2O"])
  ],
  dependencies: [
    // Dependencies declare other packages that this package depends on.
    .package(url: "https://github.com/marcrasi/penguin", .branch("patch-1")),
  ],
  targets: [
    // Targets are the basic building blocks of a package. A target can define a module or a test suite.
    // Targets can depend on other targets in this package, and on products in packages which this package depends on.
    .target(
      name: "SwiftFusion",
      dependencies: ["PenguinGraphs", "PenguinStructures"]),
    .target(
      name: "Pose2SLAMG2O",
      dependencies: ["SwiftFusion"],
      path: "Examples/Pose2SLAMG2O"),
    .testTarget(
      name: "SwiftFusionTests",
      dependencies: ["SwiftFusion"]),
  ])
