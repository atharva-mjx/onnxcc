#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <vector>

#include "onnxcc/ir/graph.h"
#include "onnxcc/ir/status.h"
#include "onnxcc/parser/parser.h"

using onnxcc::Graph;
using onnxcc::Status;

namespace {

// ── Helpers ────

// Writes bytes to a temp file and returns its path
std::filesystem::path write_temp_file(const std::vector<std::uint8_t>& bytes) {
    auto path = std::filesystem::temp_directory_path() / "onnxcc_test_parser.bin";
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()), 
            static_cast<std::streamsize>(bytes.size()));
    return path;
}

// ── parse_file tests ────

// A file that does not exist must return an error that includes the path
TEST(Parser, MissingFileReturnsError) {
    Graph g;
    Status s = onnxcc::parser::parse_file("does_not_exist.onnx", g);
    EXPECT_FALSE(s.ok());
    EXPECT_NE(std::string::npos, s.message().find("does_not_exist.onnx"));
}

// A directory path must return an error, not crash
TEST(Parser, DirectoryPathReturnsError) {
    Graph g;
    Status s = onnxcc::parser::parse_file("/tmp", g);
    EXPECT_FALSE(s.ok());
}

// ── parse_bytes tests ────

// Zero bytes is not a valid ONNX model
TEST(Parser, ZeroBytesReturnsError) {
    Graph g;
    Status s = onnxcc::parser::parse_bytes({}, g);
    EXPECT_FALSE(s.ok());
}

// Random non-ONNX bytes must not crash
TEST(Parser, NonOnnxBytesReturnsError) {
    const std::vector<std::byte> junk = {
        std::byte{0xDE}, std::byte{0xAD}, std::byte{0xBE}, std::byte{0xEF},
        std::byte{0x00}, std::byte{0xFF}, std::byte{0x12}, std::byte{0x34}
    };
    Graph g;
    Status s = onnxcc::parser::parse_bytes(junk, g);
    EXPECT_FALSE(s.ok());
}

// Truncated ONNX bytes must not crash
TEST(Parser, TruncatedBytesReturnsError) {
    // Valid protobuf prefix for a ModelProto but cut short
    const std::vector<std::byte> truncated = {
        std::byte{0x08}, std::byte{0x07},  // ir_version: 7
        std::byte{0x12}                     // start of next field, then cut
    };
    Graph g;
    Status s = onnxcc::parser::parse_bytes(truncated, g);
    EXPECT_FALSE(s.ok());
}

// A zero byte file on disk must not crash
TEST(Parser, ZeroByteFileReturnsError) {
    auto path = write_temp_file({});
    Graph g;
    Status s = onnxcc::parser::parse_file(path.string(), g);
    EXPECT_FALSE(s.ok());
    std::filesystem::remove(path);
}

// A file with random junk bytes must not crash  
TEST(Parser, JunkFileReturnsError) {
    auto path = write_temp_file({0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0xFF});
    Graph g;
    Status s = onnxcc::parser::parse_file(path.string(), g);
    EXPECT_FALSE(s.ok());
    std::filesystem::remove(path);
}

} 