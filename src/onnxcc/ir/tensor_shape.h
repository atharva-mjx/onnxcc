#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <string>
#include <compare>
#include "onnxcc/ir/datatype.h"

namespace onnxcc {

// TensorShape represents dimensions of a tensor
// With DataType it fully represents the tensor's physical layout in type-erased runtime(treat all the data as raw, untyped bytes)
// No stride member for now. Assume contiguous and row major order. Strides will be introduced for zero copy views
// Use of public members since IR is just plain data
struct TensorShape {
    std::vector<std::int64_t> dims;
    // '<' and '>' are semantically meaningless
    // defined the operator so it can be used in ordered containers
    auto operator<=>(const TensorShape&) const = default;
    // product of dims. a rank 0 shape is a scalar
    // throws std::logic_error if shape is dynamic
    // throws std::overflow_error if product exceeds size_t
    std::size_t num_elements() const;
    std::size_t num_bytes(DataType) const;
    // Formats the shape for diagnostics
    // Format specification for tests:
    // Wrapped in brackets: "[...]"
    // Elements separated by comma and space: ", "
    // Only a dim of exactly -1 (the dynamic dim) renders as "?"
    // Any other negative dim is invalid and renders as its number (e.g. "-2"), so it stays visible
    // Rank 0 renders exactly as: "[]"
    std::string to_string() const;
    // check if the shape contains any dynamic or unknown dimensions.
    // dynamic dimensions are used for dynamic sizing of batch, inputs, etc
    // onnx unknown dims are mapped to -1 by parser
    // check < 0 (not just == -1) to also catch other invalid dims, they are reported as dynamic too
    // note to_string() is narrower, it prints "?" for -1 only
    bool is_dynamic() const noexcept;
};
}