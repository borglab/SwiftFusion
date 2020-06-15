// Copyright 2019 The SwiftFusion Authors. All Rights Reserved.
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

/// An error from a failure in searching for the solution in LM iterations
public struct LevenbergMarquardtError: Swift.Error {
  public let message: String
}

/// Levenberg-Marquadt optimizer
///
/// Implements the Levenberg-Marquardt algorithm for optimizing non-linear functions.
public struct LM {
  /// The set of steps taken.
  public var step: Int = 0
  
  /// Desired precision, TODO(fan): make this actually work
  public var precision: Double = 1e-10
  
  /// Maximum number of L-M iterations
  public var max_iteration: Int = 50
  
  /// Maximum number of G-N iterations
  public var max_inner_iteration: Int = 400
  
  /// Type of the verbosity of the logging inside the optimizer
  public enum Verbosity: Int, Comparable {
    case SILENT = 0 , SUMMARY, TRYLAMBDA
    
    // Implement Comparable
    public static func < (a: Verbosity, b: Verbosity) -> Bool {
        return a.rawValue < b.rawValue
    }
  }
  
  /// Verbosity of the logging
  public var verbosity: Verbosity = .SILENT
  
  public init(precision p: Double = 1e-10, max_iteration maxiter: Int = 50) {
    self.precision = p
    self.max_iteration = maxiter
    self.step = 0
  }
  
  public mutating func optimize(graph: NewFactorGraph, initial val: inout VariableAssignments) throws {
    var old_error = graph.error(at: val)
    
    if verbosity >= .SUMMARY {
      print("[LM OUTER] initial error = \(old_error)")
    }
    
    var lambda = 1e-6
    var inner_iter_step = 0
    var inner_success = false
    
    for _ in 0..<max_iteration { // outer loop
      
      if verbosity >= .SUMMARY {
        print("[LM OUTER] outer loop start, error = \(graph.error(at: val))")
      }
      
      let gfg = graph.linearized(at: val)
      let dx = val.tangentVectorZeros
      
      // Try lambda steps
      while true {
        if verbosity >= .TRYLAMBDA {
          print("[LM INNER] starting one iteration, lambda = \(lambda)")
        }
        
        var damped = gfg
        
        damped.addScalarJacobians(lambda)
        
        let old_linear_error = damped.errorVectors(at: dx).squaredNorm
        
        var dx_t = dx
        var optimizer = GenericCGLS(precision: 0, max_iteration: 200)
        optimizer.optimize(gfg: damped, initial: &dx_t)
        if verbosity >= .TRYLAMBDA {
          print("[LM INNER] damped error = \(damped.errorVectors(at: dx_t).squaredNorm), lambda = \(lambda)")
        }
        let oldval = val
        val.move(along: -1 * dx_t)
        let this_error = graph.error(at: val)
        let delta_error = old_error - this_error
        
        if verbosity >= .TRYLAMBDA {
          print("[LM INNER] nonlinear error = \(this_error), delta error = \(delta_error)")
        }
        
        let new_linear_error = damped.errorVectors(at: dx_t).squaredNorm
        let model_fidelity = delta_error / (old_linear_error - new_linear_error)
        
        if verbosity >= .TRYLAMBDA {
          print("[LM INNER] linear error = \(new_linear_error), delta error = \(old_linear_error - new_linear_error)")
          print("[LM INNER] model fidelity = \(model_fidelity)")
        }
        
        inner_success = false
        if delta_error > .ulpOfOne && model_fidelity > 0.01 {
          old_error = this_error
          
          // Success, decrease lambda
          if lambda > 1e-10 {
            lambda = lambda / 10
          } else {
            break
          }
          inner_success = true
        } else {
          if verbosity >= .TRYLAMBDA {
            print("[LM INNER] fail, trying to increase lambda")
          }
          // increase lambda and retry
          val = oldval
          if lambda > 1e20 {
            if verbosity >= .TRYLAMBDA {
              print("[LM INNER] giving up in lambda search")
            }
            throw LevenbergMarquardtError(message: "maximum lambda reached, giving up")
          }
          lambda = lambda * 10
        }
        
        inner_iter_step += 1
        if inner_iter_step > 5 || inner_success {
          break
        }
      }
      
      step += 1
    }
    
    if verbosity >= .SUMMARY {
      print("[FINAL   ] final error = \(graph.error(at: val))")
    }
  }
}
