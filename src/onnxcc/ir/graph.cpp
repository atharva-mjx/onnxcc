#include "onnxcc/ir/graph.h"

#include <string_view>
#include <utility>

namespace onnxcc {

namespace {

// we write the moves by hand instead of = default because when we move a graph we steal its content, vector and string
// are moved and the old one is left empty, but plain numbers like ir_version and opset_version are only copied,
// so the old graph would still have them, and graph.h says a moved from graph must be empty,
// so the function clear_graph_contents is used to clear the source graph after move.
void clear_graph_contents(Graph& g) noexcept {
    g.name.clear();
    g.ir_version = 0;
    g.opset_version = 0;
    g.inputs.clear();
    g.outputs.clear();
    g.initializers.clear();
    g.value_info.clear();
    g.nodes.clear();
}

const Tensor* find_by_name(const std::vector<Tensor>& tensors, std::string_view name) {
    for (const Tensor& t : tensors) {
        if (t.name == name) {
            return &t;
        }
    }
    return nullptr;
}

}  // namespace

// both moves are noexcept so std::vector<Graph> can move its elements safely when it grows
// this is the move constructor, it moves the content of other graph to a newly built graph that is empty, and then clear the other graph.
Graph::Graph(Graph&& other) noexcept
    : name(std::move(other.name)),
      ir_version(other.ir_version),
      opset_version(other.opset_version),
      inputs(std::move(other.inputs)),
      outputs(std::move(other.outputs)),
      initializers(std::move(other.initializers)),
      value_info(std::move(other.value_info)),
      nodes(std::move(other.nodes)) {
      clear_graph_contents(other);
}

// this is the move assignment operator, it moves the content of other graph to already existing graph by releasing its data and moving the
// content, and then clear the other graph.
Graph& Graph::operator=(Graph&& other) noexcept {
    if (this != &other) {
        name = std::move(other.name);
        ir_version = other.ir_version;
        opset_version = other.opset_version;
        inputs = std::move(other.inputs);
        outputs = std::move(other.outputs);
        initializers = std::move(other.initializers);
        value_info = std::move(other.value_info);
        nodes = std::move(other.nodes);
        clear_graph_contents(other);
    }
    return *this;
}

const Tensor* Graph::find_initializer(std::string_view tensor_name) const {
    // empty name = absent optional input (node.h), never a real initializer
    if (tensor_name.empty()) {
        return nullptr;
    }
    return find_by_name(initializers, tensor_name);
}

const Tensor* Graph::find_value_info(std::string_view tensor_name) const {
    return find_by_name(value_info, tensor_name);
}

}  // namespace onnxcc