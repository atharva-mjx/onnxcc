#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "onnxcc/ir/graph.h"

namespace onnxcc {

// these are the 4 questions compiler ask while compiling the code, and if anyone of them is false, compiler stop the build with an error.
static_assert(!std::is_copy_constructible_v<Graph>);
static_assert(!std::is_copy_assignable_v<Graph>);
static_assert(std::is_nothrow_move_constructible_v<Graph>);
static_assert(std::is_nothrow_move_assignable_v<Graph>);

namespace {

Tensor make_tensor(const std::string& name, DataType dtype = DataType::FLOAT32,
                   bool is_initializer = false) {
    Tensor t;
    t.name = name;
    t.dtype = dtype;
    t.is_initializer = is_initializer;
    return t;
}


Node make_node(const std::string& name, const std::string& op_type) {
    Node n;
    n.name = name;
    n.op_type = op_type;
    return n;
}

Graph make_graph() {
    Graph g;
    g.name = "tiny_mlp";
    g.ir_version = 8;
    g.opset_version = 13;
    g.inputs.push_back(make_tensor("input"));
    g.outputs.push_back(make_tensor("output"));

    Tensor w1 = make_tensor("w1", DataType::FLOAT32, true);
    w1.data = {1, 2, 3, 4};
    g.initializers.push_back(std::move(w1));

    Tensor b1 = make_tensor("b1", DataType::FLOAT32, true);
    b1.data = {5, 6, 7, 8};
    g.initializers.push_back(std::move(b1));

    g.value_info.push_back(make_tensor("hidden"));

    Node matmul = make_node("matmul_1", "MatMul");
    matmul.inputs = {"input", "w1"};
    matmul.outputs = {"hidden"};
    g.nodes.push_back(std::move(matmul));

    Node relu = make_node("relu_1", "Relu");
    relu.inputs = {"hidden"};
    relu.outputs = {"output"};
    g.nodes.push_back(std::move(relu));

    return g;
}

// this function is used to check if the source graph is empty after move.
void expect_empty(const Graph& g) {
    EXPECT_TRUE(g.name.empty());
    EXPECT_EQ(g.ir_version, 0);
    EXPECT_EQ(g.opset_version, 0);
    EXPECT_TRUE(g.inputs.empty());
    EXPECT_TRUE(g.outputs.empty());
    EXPECT_TRUE(g.initializers.empty());
    EXPECT_TRUE(g.value_info.empty());
    EXPECT_TRUE(g.nodes.empty());
}

// this function is used to check if new graph has all the content after move.
void expect_full(const Graph& g) {
    EXPECT_EQ(g.name, "tiny_mlp");
    EXPECT_EQ(g.ir_version, 8);
    EXPECT_EQ(g.opset_version, 13);

    ASSERT_EQ(g.inputs.size(), 1u);
    EXPECT_EQ(g.inputs[0].name, "input");
    EXPECT_FALSE(g.inputs[0].is_initializer);

    ASSERT_EQ(g.outputs.size(), 1u);
    EXPECT_EQ(g.outputs[0].name, "output");

    const std::vector<std::uint8_t> w1_bytes{1, 2, 3, 4};
    const std::vector<std::uint8_t> b1_bytes{5, 6, 7, 8};
    ASSERT_EQ(g.initializers.size(), 2u);
    EXPECT_EQ(g.initializers[0].name, "w1");
    EXPECT_EQ(g.initializers[0].dtype, DataType::FLOAT32);
    EXPECT_TRUE(g.initializers[0].is_initializer);
    EXPECT_EQ(g.initializers[0].data, w1_bytes);
    EXPECT_EQ(g.initializers[1].name, "b1");
    EXPECT_TRUE(g.initializers[1].is_initializer);
    EXPECT_EQ(g.initializers[1].data, b1_bytes);

    ASSERT_EQ(g.value_info.size(), 1u);
    EXPECT_EQ(g.value_info[0].name, "hidden");

    const std::vector<std::string> matmul_in{"input", "w1"};
    const std::vector<std::string> matmul_out{"hidden"};
    const std::vector<std::string> relu_in{"hidden"};
    const std::vector<std::string> relu_out{"output"};
    ASSERT_EQ(g.nodes.size(), 2u);
    EXPECT_EQ(g.nodes[0].name, "matmul_1");
    EXPECT_EQ(g.nodes[0].op_type, "MatMul");
    EXPECT_EQ(g.nodes[0].inputs, matmul_in);
    EXPECT_EQ(g.nodes[0].outputs, matmul_out);
    EXPECT_EQ(g.nodes[1].name, "relu_1");
    EXPECT_EQ(g.nodes[1].op_type, "Relu");
    EXPECT_EQ(g.nodes[1].inputs, relu_in);
    EXPECT_EQ(g.nodes[1].outputs, relu_out);
}

}  // namespace

TEST(Graph, FindInitializerHit) {
    const Graph g = make_graph();
    const Tensor* t = g.find_initializer("b1");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->name, "b1");
    EXPECT_EQ(t, &g.initializers[1]);
}

TEST(Graph, FindInitializerMiss) {
    const Graph g = make_graph();
    EXPECT_EQ(g.find_initializer("nope"), nullptr);
    EXPECT_EQ(g.find_initializer(""), nullptr);
    EXPECT_EQ(g.find_initializer("w"), nullptr);
}

TEST(Graph, FindValueInfoHit) {
    const Graph g = make_graph();
    const Tensor* t = g.find_value_info("hidden");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->name, "hidden");
    EXPECT_EQ(t, &g.value_info[0]);
}

TEST(Graph, FindValueInfoMiss) {
    const Graph g = make_graph();
    EXPECT_EQ(g.find_value_info("nope"), nullptr);
}

TEST(Graph, LookupsOnEmptyGraph) {
    const Graph g;
    EXPECT_EQ(g.find_initializer("w1"), nullptr);
    EXPECT_EQ(g.find_value_info("hidden"), nullptr);
}

TEST(Graph, LookupsSearchOnlyTheirOwnVector) {
    const Graph g = make_graph();
    EXPECT_EQ(g.find_value_info("w1"), nullptr);
    EXPECT_EQ(g.find_initializer("hidden"), nullptr);
    EXPECT_EQ(g.find_initializer("input"), nullptr);
}

TEST(Graph, MoveConstructTransfersEverything) {
    Graph a = make_graph();
    Graph b(std::move(a));
    expect_full(b);
    expect_empty(a);  
}

TEST(Graph, MoveAssignTransfersEverything) {
    Graph a = make_graph();
    Graph b;
    b.name = "old";
    b.initializers.push_back(make_tensor("stale"));

    b = std::move(a);

    expect_full(b);
    expect_empty(a);  
    EXPECT_EQ(b.find_initializer("stale"), nullptr);
}

TEST(Graph, LookupsWorkAfterMove) {
    Graph a = make_graph();
    Graph b = std::move(a);
    EXPECT_NE(b.find_initializer("w1"), nullptr);
    EXPECT_NE(b.find_value_info("hidden"), nullptr);
    EXPECT_EQ(a.find_initializer("w1"), nullptr);  
}

TEST(Graph, SelfMoveAssignKeepsTheGraph) {
    Graph a = make_graph();
    Graph& alias = a;
    a = std::move(alias);
    expect_full(a);
}

TEST(Graph, SurvivesVectorGrowth) {
    std::vector<Graph> graphs;
    for (int i = 0; i < 20; ++i) {
        Graph g = make_graph();
        g.name = "g" + std::to_string(i);
        graphs.push_back(std::move(g));
    }
    ASSERT_EQ(graphs.size(), 20u);
    EXPECT_EQ(graphs[0].name, "g0");
    EXPECT_EQ(graphs[19].name, "g19");
    EXPECT_NE(graphs[7].find_initializer("w1"), nullptr);
}

TEST(Graph, FindInitializerEmptyNameReturnsNullptr) {
    Graph g = make_graph();
    g.initializers.push_back(make_tensor(""));
    EXPECT_EQ(g.find_initializer(""), nullptr);
    EXPECT_NE(g.find_initializer("w1"), nullptr);
}

}  // namespace onnxcc