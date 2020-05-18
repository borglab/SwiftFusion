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

import Foundation
#if !os(macOS)
import FoundationNetworking
#endif

/// Dictionary from dataset id to dataset url.
fileprivate let datasets = [
  "input_INTEL_g2o.txt":
    "https://github.com/pkchen1129/Mobile-Robotics/raw/8551df10ba8af36801403daeba710c1f9c9e54cd/ps4/code/dataset/input_INTEL_g2o.txt"
]

/// Returns a dataset cached on the local system.
///
/// If the dataset with `id` is available on the local system, returns it immediately.
/// Otherwise, downloads the dataset to the cache and then returns it.
///
/// - Parameter `cacheDirectory`: The directory where the cached datasets are stored.
public func cachedDataset(
  _ id: String,
  cacheDirectory: URL = URL(fileURLWithPath: NSHomeDirectory())
    .appendingPathComponent(".SwiftFusionDatasetCache", isDirectory: true)
) throws -> URL {
  try createDirectoryIfMissing(at: cacheDirectory.path)
  let cacheEntry = cacheDirectory.appendingPathComponent(id)
  if FileManager.default.fileExists(atPath: cacheEntry.path) {
    return cacheEntry
  }
  guard let url = datasets[id] else {
    throw DatasetCacheError(message: "No such dataset: \(id)")
  }
  print("Downloading \(url) to \(cacheEntry)")
  guard let source = URL(string: url) else {
    throw DatasetCacheError(message: "Could not parse URL: \(url)")
  }
  let data = try Data.init(contentsOf: source)
  try data.write(to: cacheEntry)
  print("Downloaded \(cacheEntry)!")
  return cacheEntry
}

/// An error from getting a cached dataset.
public struct DatasetCacheError: Swift.Error {
  public let message: String
}

/// Creates a directory at a path, if missing. If the directory exists, this does nothing.
///
/// - Parameters:
///   - path: The path of the desired directory.
fileprivate func createDirectoryIfMissing(at path: String) throws {
    guard !FileManager.default.fileExists(atPath: path) else { return }
    try FileManager.default.createDirectory(
        atPath: path,
        withIntermediateDirectories: true,
        attributes: nil)
}
