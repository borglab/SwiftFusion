import Foundation

/// Utilities for timing and couting regions of code.
///
/// NOT threadsafe, which we should probably fix soon.

fileprivate var startedTimers: [String: UInt64] = [:]
fileprivate var accumulatedTimers: [String: UInt64] = [:]

/// Start the `name` timer.
///
/// Precondition: The `name` timer is not running.
public func startTimer(_ name: String) {
//  guard startedTimers[name] == nil else { preconditionFailure("timer \(name) is already started") }
//  startedTimers[name] = DispatchTime.now().uptimeNanoseconds
}

/// Stop the `name` timer.
///
/// Precondition: The `name` timer is running.
public func stopTimer(_ name: String) {
//  guard let start = startedTimers[name] else { preconditionFailure("timer \(name) is not running") }
//  startedTimers[name] = nil
//  accumulatedTimers[name, default: 0] += DispatchTime.now().uptimeNanoseconds - start
}

/// Print the total times accumulated for each timer.
public func printTimers() {
//  guard startedTimers.count == 0 else { preconditionFailure("timers are still running: \(startedTimers)") }
//  for (name, duration) in accumulatedTimers {
//    print("\(name): \(Double(duration) / 1e9) seconds")
//  }
}

fileprivate var counters: [String: Int] = [:]

/// Increment the `name` counter.
public func incrementCounter(_ name: String) {
//  counters[name, default: 0] += 1
}

/// Print the total counts for each counter.
public func printCounters() {
//  for (name, count) in counters {
//    print("\(name): \(count)")
//  }
}
