#pragma once

#include <cstddef>
#include <span>
#include <string_view>

#include "onnxcc/ir/graph.h"
#include "onnxcc/ir/status.h"

namespace onnxcc::parser {

// Reads an ONNX model from a file on disk and fills out.
// Returns an error Status if:
//   - the file does not exist (path is included in the error message)
//   - the file is not a regular file (e.g. a directory)
//   - the bytes are not a valid ONNX model
//   - the model uses an unsupported opset or data type
Status parse_file(std::string_view path, Graph& out);

// Reads an ONNX model from raw bytes already in memory.
// Useful for testing with truncated, zero-byte, or crafted inputs.
// Returns an error Status if the bytes are not a valid ONNX model.
Status parse_bytes(std::span<const std::byte> bytes, Graph& out);

} 