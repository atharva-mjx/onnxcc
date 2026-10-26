#include "onnxcc/parser/parser.h"
#include "onnxcc/parser/parser_internal.h"

#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <google/protobuf/message_lite.h>
#include <onnx/onnx_pb.h>

#include "onnxcc/ir/attribute.h"
#include "onnxcc/ir/datatype.h"
#include "onnxcc/ir/graph.h"
#include "onnxcc/ir/node.h"
#include "onnxcc/ir/status.h"
#include "onnxcc/ir/tensor.h"
#include "onnxcc/ir/tensor_shape.h"

namespace onnxcc::parser {

// Verify at startup that the runtime protobuf library matches
// the headers we compiled against. This is a no-op if versions match,
// and aborts with a clear message if they don't.
// ShutdownProtobufLibrary() is intentionally called here via atexit
// rather than a destructor to avoid ordering issues with static objects.
namespace {

void shutdown_protobuf() {
    google::protobuf::ShutdownProtobufLibrary();
}

// Runs before main via static initialisation
const bool kProtobufInit = []() {
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    std::atexit(shutdown_protobuf);
    return true;
}();

} // namespace

// ── parse_file ──

Status parse_file(std::string_view path, Graph& out) {
    // Open in binary mode — ONNX files are binary protobuf
    std::ifstream file{std::string(path), std::ios::binary};
    if (!file.is_open()) {
        return Status::error(std::format("cannot open model file '{}'", path));
    }

    // Read entire file into memory
    std::vector<char> buffer{
        std::istreambuf_iterator<char>{file},
        std::istreambuf_iterator<char>{}
    };

    if (!file.good() && !file.eof()) {
        return Status::error(std::format("failed to read model file '{}'", path));
    }

    return parse_bytes(
        std::span<const std::byte>{
            reinterpret_cast<const std::byte*>(buffer.data()),
            buffer.size()
        },
        out
    );
}

// ── parse_bytes ──

Status parse_bytes(std::span<const std::byte> bytes, Graph& out) {
    // An empty span is not a valid ONNX model
    if (bytes.empty()) {
        return Status::error("model is empty (zero bytes)");
    }

    onnx::ModelProto model;
    const bool ok = model.ParseFromArray(
        bytes.data(),
        static_cast<int>(bytes.size())
    );
    if (!ok) {
        return Status::error("failed to parse bytes as a valid ONNX model");
    }

    return internal::model_to_graph(&model, out);
}

} 


// ── internal: model_to_graph ──

namespace onnxcc::parser::internal {

namespace {

// Maps an ONNX TensorProto::DataType integer code to our DataType enum.
// Returns DataType::UNDEFINED for any code we do not support.
DataType map_dtype(int onnx_code) {
    switch (onnx_code) {
        case 1:  return DataType::FLOAT32;
        case 6:  return DataType::INT32;
        case 7:  return DataType::INT64;
        case 9:  return DataType::BOOL;
        case 11: return DataType::FLOAT64;
        default: return DataType::UNDEFINED;
    }
}

// Converts an ONNX TensorShapeProto to our TensorShape.
// Unknown/dynamic dimensions are mapped to -1 as documented in tensor_shape.h.
TensorShape map_shape(const onnx::TensorShapeProto& proto) {
    TensorShape shape;
    for (const auto& dim : proto.dim()) {
        shape.dims.push_back(
            dim.has_dim_value() ? dim.dim_value() : -1
        );
    }
    return shape;
}

// Converts an ONNX TensorProto (initializer/weight) to our Tensor.
Status map_initializer(const onnx::TensorProto& proto, Tensor& out) {
    out.name  = proto.name();
    out.dtype = map_dtype(proto.data_type());

    if (out.dtype == DataType::UNDEFINED) {
        return Status::error(std::format(
            "initializer '{}' uses unsupported dtype code {}",
            proto.name(), proto.data_type()
        ));
    }

    for (const std::int64_t d : proto.dims()) {
        out.shape.dims.push_back(d);
    }

    // ONNX stores initializer data either as raw_data (preferred, always binary)
    // or as typed repeated fields (float_data, int32_data, etc.).
    // We only handle raw_data here; typed fields are used by older models.
    if (proto.has_raw_data()) {
        const std::string& raw = proto.raw_data();
        out.data.assign(
            reinterpret_cast<const std::uint8_t*>(raw.data()),
            reinterpret_cast<const std::uint8_t*>(raw.data()) + raw.size()
        );
    }

    out.is_initializer = true;
    return Status::ok_value();
}

// Converts an ONNX ValueInfoProto (input, output, or value_info entry to our Tensor.
// Only shape and dtype are filled ,no data.
Tensor map_value_info(const onnx::ValueInfoProto& proto) {
    Tensor t;
    t.name = proto.name();

    if (proto.type().has_tensor_type()) {
        const auto& tt = proto.type().tensor_type();
        t.dtype = map_dtype(tt.elem_type());
        if (tt.has_shape()) {
            t.shape = map_shape(tt.shape());
        }
    }

    return t;
}

// Converts an ONNX AttributeProto to our Attribute variant.
// Returns std::nullopt for attribute types we do not support yet.
std::optional<Attribute> map_attribute(const onnx::AttributeProto& proto) {
    switch (proto.type()) {
        case onnx::AttributeProto::INT:
            return Attribute{proto.i()};
        case onnx::AttributeProto::FLOAT:
            return Attribute{proto.f()};
        case onnx::AttributeProto::STRING:
            return Attribute{proto.s()};
        case onnx::AttributeProto::INTS:
            return Attribute{std::vector<std::int64_t>{
                proto.ints().begin(), proto.ints().end()
            }};
        case onnx::AttributeProto::FLOATS:
            return Attribute{std::vector<float>{
                proto.floats().begin(), proto.floats().end()
            }};
        default:
            // TENSOR, GRAPH, SPARSE_TENSOR etc. are not supported yet
            return std::nullopt;
    }
}

// Converts an ONNX NodeProto to our Node.
// Auto-generates a name if the ONNX file omits it,node.h guarantees
// the name field is always non-empty.
Node map_node(const onnx::NodeProto& proto, int index) {
    Node node;

    node.name    = proto.name().empty()
                       ? std::format("node_{}", index)
                       : proto.name();
    node.op_type = proto.op_type();

    for (const auto& in  : proto.input())  node.inputs.push_back(in);
    for (const auto& out : proto.output()) node.outputs.push_back(out);

    // Map each attribute; silently skip unsupported types
    for (const auto& attr_proto : proto.attribute()) {
        auto maybe = map_attribute(attr_proto);
        if (maybe.has_value()) {
            node.attributes.emplace(attr_proto.name(), std::move(*maybe));
        }
    }

    return node;
}

} 

// ── model_to_graph ──

Status model_to_graph(const void* model_proto_ptr, Graph& out) {
    const auto& model = *static_cast<const onnx::ModelProto*>(model_proto_ptr);

    if (!model.has_graph()) {
        return Status::error("ONNX model has no graph");
    }

    out.ir_version = model.ir_version();

    // Read the opset version from the empty-string domain (the standard
    // ONNX domain). Do NOT use opset_import(0),index order is not
    // guaranteed. Both "" and "ai.onnx" refer to the standard domain.
    for (const auto& opset : model.opset_import()) {
        if (opset.domain().empty() || opset.domain() == "ai.onnx") {
            out.opset_version = opset.version();
            break;
        }
    }

    const onnx::GraphProto& graph = model.graph();
    out.name = graph.name();

    for (const auto& vi   : graph.input())      out.inputs.push_back(map_value_info(vi));
    for (const auto& vi   : graph.output())     out.outputs.push_back(map_value_info(vi));
    for (const auto& vi   : graph.value_info()) out.value_info.push_back(map_value_info(vi));

    for (const auto& init : graph.initializer()) {
        Tensor t;
        Status s = map_initializer(init, t);
        if (!s.ok()) return s;
        out.initializers.push_back(std::move(t));
    }

    int index = 0;
    for (const auto& node_proto : graph.node()) {
        out.nodes.push_back(map_node(node_proto, index++));
    }

    return Status::ok_value();
}

} 