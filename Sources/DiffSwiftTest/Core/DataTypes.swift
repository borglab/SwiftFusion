// /// A scalar data type compatible with SwiftFusion.
// ///
// /// Types that conform to `SwiftFusionScalar` can be used as the `Scalar` associated type of
// /// `Tensor`.
// //
// // This includes all `_SwiftFusionDataTypeCompatible` types except `String`.
// public protocol SwiftFusionScalar {}

// public typealias SwiftFusionNumeric = SwiftFusionScalar & Numeric
// public typealias SwiftFusionSignedNumeric = SwiftFusionScalar & SignedNumeric
// public typealias SwiftFusionInteger = SwiftFusionScalar & BinaryInteger

// /// A floating-point data type that conforms to `Differentiable` and is compatible with SwiftFusion.
// ///
// /// - Note: `Tensor` conditionally conforms to `Differentiable` when the `Scalar` associated type
// ///   conforms `SwiftFusionFloatingPoint`.
// public protocol SwiftFusionFloatingPoint:
//     SwiftFusionScalar & BinaryFloatingPoint & Differentiable & ElementaryFunctions
//     where Self.RawSignificand: FixedWidthInteger,
//           Self == Self.TangentVector {}

// extension Float: SwiftFusionFloatingPoint {}
// extension Double: SwiftFusionFloatingPoint {}

// extension Bool: SwiftFusionScalar {}

// extension Int8: SwiftFusionScalar {}

// extension UInt8: SwiftFusionScalar {}

// extension Int16: SwiftFusionScalar {}

// extension UInt16: SwiftFusionScalar {}

// extension Int32: SwiftFusionScalar {}

// extension UInt32: SwiftFusionScalar {}

// extension Int64: SwiftFusionScalar {}

// extension UInt64: SwiftFusionScalar {}

// @frozen
// public struct BFloat16 {
//     @usableFromInline var data: Int16 = 0
//     private init() {}
// }

// extension BFloat16: SwiftFusionScalar {}

// extension Float: SwiftFusionScalar {}

// extension Double: SwiftFusionScalar {}