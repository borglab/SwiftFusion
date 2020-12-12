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

import _Differentiation
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
  
  /// Maximam Lambda
  public var max_lambda: Double = 1e32
  
  /// Minimum Lambda
  public var min_lambda: Double = 1e-16
  
  /// Initial Lambda
  public var initial_lambda: Double = 1e-4
  
  /// Lambda Factor
  public var lambda_factor: Double = 2
  
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
  
  public mutating func optimize(graph: FactorGraph, initial val: inout VariableAssignments,
                                hook: ((FactorGraph, VariableAssignments, Double, Int) -> Void)? = nil) throws {
    var old_error = graph.linearizableError(at: val)
    
    if verbosity >= .SUMMARY {
      print("[LM OUTER] initial error = \(old_error)")
    }
    
    var lambda: Double = initial_lambda
    var inner_iter_step = 0
    var inner_success = false
    var all_done = false
    
    for _ in 0..<max_iteration { // outer loop
      // Do logging first
      if let h = hook {
        h(graph, val, lambda, step)
      }
      
      if verbosity >= .SUMMARY {
        print("[LM OUTER] outer loop start, error = \(graph.linearizableError(at: val))")
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
        
        let old_linear_error = damped.error(at: dx)
        
        var dx_t = dx
        var optimizer = GenericCGLS(precision: 1e-10, max_iteration: max_inner_iteration)
        optimizer.optimize(gfg: damped, initial: &dx_t)
        if verbosity >= .TRYLAMBDA {
          print("[LM INNER] damped error = \(damped.error(at: dx_t)), lambda = \(lambda)")
        }
        let oldval = val
        val.move(along: dx_t)
        let this_error = graph.linearizableError(at: val)
        let delta_error = old_error - this_error
        
        if verbosity >= .TRYLAMBDA {
          print("[LM INNER] nonlinear error = \(this_error), delta error = \(delta_error)")
        }
        
        let new_linear_error = damped.error(at: dx_t)
        let model_fidelity = delta_error / (old_linear_error - new_linear_error)
        
        if verbosity >= .TRYLAMBDA {
          print("[LM INNER] linear error = \(new_linear_error), delta error = \(old_linear_error - new_linear_error)")
          print("[LM INNER] model fidelity = \(model_fidelity)")
        }
        
        inner_success = false
        if delta_error > .ulpOfOne && model_fidelity > 0.01 {
          old_error = this_error
          
          // Success, decrease lambda
          if lambda > min_lambda {
            lambda = lambda / lambda_factor
          }
          
          inner_success = true
        } else {
          if verbosity >= .TRYLAMBDA {
            print("[LM INNER] fail, trying to increase lambda or give up")
          }
          
          // increase lambda and retry
          val = oldval
          if lambda > max_lambda {
            if verbosity >= .TRYLAMBDA {
              print("[LM INNER] giving up in lambda search")
            }
            throw LevenbergMarquardtError(message: "maximum lambda reached, giving up")
          }
          lambda = lambda * lambda_factor
        }
        
        if model_fidelity > 0.5 && delta_error < precision || this_error < precision {
          if verbosity >= .SUMMARY {
            print("[LM INNER] reached the target precision, exiting")
          }
          inner_success = true
          all_done = true
          break
        }
        
        inner_iter_step += 1
        if inner_success {
          break
        }
      }
      
      step += 1
      
      if all_done {
        // Log before exit
        if let h = hook {
          h(graph, val, lambda, step)
        }
        
        break
      }
    }
    
    if verbosity >= .SUMMARY {
      print("[FINAL   ] final error = \(graph.error(at: val))")
    }
  }
}
