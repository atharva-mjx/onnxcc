#pragma once

#include "onnxcc/ir/graph.h"
#include "onnxcc/ir/status.h"

// Internal contract between parser stages.
// parser.cpp uses protobuf directly and is the ONLY file that includes
// <onnx/onnx_pb.h>. This header deliberately avoids protobuf types so future tasks never need to link against protobuf directly.

namespace onnxcc::parser::internal {

// Converts a decoded protobuf ModelProto into the Graph IR.
// model_proto is passed as const void* to keep protobuf types out ofthis header. 
// parser.cpp casts it back to const onnx::ModelProto*.
// Returns an error Status if:
//   - the model has no graph
//   - a tensor uses an unsupported DataType
//   - a required field is missing

Status model_to_graph(const void* model_proto, Graph& out);

} 