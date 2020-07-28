// Copyright 2020 The SwiftFusion Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Datasets
import Foundation

/// Downloads the bee dataset (if it's not already present), and returns its URL on the local
/// system.
internal func downloadBeeDatasetIfNotPresent() -> URL {
  let downloadDir = DatasetUtilities.defaultDirectory.appendingPathComponent(
    "bees_v2", isDirectory: true)
  let directoryExists = FileManager.default.fileExists(atPath: downloadDir.path)
  let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadDir.path)
  let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

  guard !directoryExists || directoryEmpty else { return downloadDir }

  let remoteRoot = URL(
    string: "https://storage.googleapis.com/swift-tensorflow-misc-files/beetracking")!

  let _ = DatasetUtilities.downloadResource(
    filename: "beedata_v2", fileExtension: "tar.gz",
    remoteRoot: remoteRoot, localStorageDirectory: downloadDir
  )

  return downloadDir
}
